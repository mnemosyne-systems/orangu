// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Compiling linear layers for the NPU, through ONNX Runtime's Zhouyi
//! execution provider.
//!
//! [`crate::npu`] *runs* graphs. This module *builds* one: it emits a small
//! ONNX model for a single linear projection, hands it to ONNX Runtime with
//! the Zhouyi provider attached, and the provider compiles it for the NPU.
//! What comes back is an AIPU executable that [`crate::npu`] can load and
//! run as many times as you like.
//!
//! The split is the point. Compiling costs a few hundred milliseconds;
//! executing costs a few hundred *micro*seconds. So compile once, keep the
//! bytes, and run them through the runtime:
//!
//! The two halves belong in **separate processes** — see below, this is not
//! a preference. In the one that compiles:
//!
//! ```no_run
//! # use orangu::npu_ort;
//! # let (weights, k, n, tokens) = (vec![0.0f32; 4], 2usize, 2usize, 1usize);
//! let compiler = npu_ort::NpuOrt::open().expect("ONNX Runtime");
//! let layer = compiler.compile_linear(&weights, k, n, tokens, (-4.0, 4.0))?;
//! std::fs::write("q_proj.npu", layer.to_bytes())?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! and in the one that serves:
//!
//! ```no_run
//! # use orangu::{npu, npu_ort};
//! # let (x, tokens) = (vec![0.0f32; 4], 1usize);
//! let layer = npu_ort::CompiledLinear::from_bytes(&std::fs::read("q_proj.npu")?)?;
//! let runtime = npu::NpuRuntime::open().expect("NPU");
//! let layer = layer.into_bound(&runtime)?; // the driver keeps its own copy
//! let mut out = Vec::new();
//! for _ in 0..1000 {
//!     layer.forward_into(&x, tokens, &mut out)?; // ~0.7 ms, allocation-free
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Why ONNX Runtime and not the vendor graph API directly
//!
//! `libcompass_runtime_midware` exposes graph construction as plain C, and
//! it does work as far as compiling a graph — but it has no way to declare
//! a *runtime input*: every node builder requires a non-null data pointer,
//! which makes the tensor a baked constant, and the `INPUT` operation the
//! compiler uses internally is not reachable through `add_node`. Its
//! structures are also undocumented, recovered only by disassembly. ONNX
//! Runtime's C API is public and version-stable, and its provider already
//! solves graph inputs, quantization and connectivity. The trade is more
//! code here against no undocumented ABI.
//!
//! # Why a 1x1 convolution rather than a matmul
//!
//! `MatMul` and `Gemm` are both accepted by the provider and both lower
//! through `fuse_constant_matmul_to_fc`, whose `FullyConnected` subnodes the
//! Zhouyi compiler's own subgraph builder then rejects
//! (`subgraphop.cpp:718`, "Unsupport node type: FullyConnected"). That
//! reproduces at every shape tried, from 4x8x16 to 32x128x128, so it is a
//! defect in the shipped stack rather than a limit of any one model.
//!
//! A 1x1 convolution computes the same thing and routes through `CONV_2D`
//! on the fixed-function engine instead, which compiles and runs. Mapping
//! a linear layer to a 1x1 convolution is the conventional approach on this
//! class of accelerator, so this is the normal path rather than a
//! workaround. It also needs no external assembler: the `MatMul` route
//! shells out to `aipuoas`, which the vendor BSP ships only as a library
//! symbol, whereas the convolution route does not.
//!
//! # Why the compiled bytes are extracted rather than the session used
//!
//! A Zhouyi session runs **once**. The second `Run` on the same session
//! never returns — verified against the CPU provider, which repeats happily
//! on the identical model, and reproduced again through a completely
//! separate session-construction path, so it is the provider rather than
//! ONNX Runtime or this module. The defect is not even at the ONNX Runtime
//! layer: driving `libcompass_runtime_midware` directly, with no ONNX
//! Runtime involved, the second `execute` blocks in `poll()` waiting on a
//! completion that never arrives.
//!
//! So this module does not execute through ONNX Runtime at all. It asks the
//! provider to compile, takes the compiled executable, and hands it to
//! [`crate::npu`], which reuses a job and runs the same graph indefinitely.
//! Measured on a 512x512 projection over 128 tokens: ~110 ms to compile,
//! ~1 ms to load, ~0.69 ms per inference thereafter. Three such layers fused
//! into one graph by [`NpuOrt::compile_stack`] run in 0.68 ms — 295 GFLOP/s,
//! and 2.75x what the same three cost as separate graphs.
//!
//! Extraction uses ONNX Runtime's own EPContext mechanism — the standard
//! way a provider caches a compiled model. With `ep.context_enable` set,
//! session creation writes an ONNX file whose single `EPContext` node
//! carries the compiled binary in its `ep_cache_context` attribute. For
//! this provider that binary is a bare AIPU executable, which is exactly
//! what [`crate::npu::NpuRuntime::load_graph_bytes`] wants.
//!
//! # Compile and serve must be separate processes
//!
//! The two stacks reach the same device through different userspace
//! libraries, and they cannot share a process — in either direction.
//!
//! The cause is concrete: `libnoe.so.0` carries its own copy of the vendor
//! UMD, and 342 of its symbols collide with `libaipu_driver.so`, which the
//! provider pulls in. Whichever loads first with `RTLD_GLOBAL` wins every
//! one of those bindings, so one library's code ends up calling the other's
//! implementation of `aipudrv::MemoryBase` and friends — two builds sharing
//! one object layout. The symptoms are what that predicts: opening NOE
//! first makes the provider's graph preparation fail with "Job size
//! provided is an invalid one", while compiling first leaves NOE running a
//! graph once and then returning `NOE_STATUS_ERROR_TIMEOUT` — but only at
//! some shapes, which is exactly how a violation like this presents.
//! Loading the provider with `RTLD_LOCAL` to isolate it does not rescue
//! this either; NOE then fails to load the graph at all.
//!
//! So: compile in one process, serve in another. Use [`CompiledLinear::to_bytes`]
//! to persist the result and [`CompiledLinear::from_bytes`] to pick it back
//! up. That is a better shape regardless — compiling costs a thousand times
//! what an inference does, so it belongs in a build step whose output is
//! cached, not on a serving path.
//!
//! A process that has opened [`NpuOrt`] must not go on to open
//! [`crate::npu::NpuRuntime`]. [`CompiledLinear::bind`] is for the serving
//! side, where the artifact was read from bytes and no compiler was ever
//! opened.
//!
//! # Why everything is quantized
//!
//! The hardware is an integer engine. A float model is refused outright —
//! "input 0th tensor doesn't support operand type: FLOAT32" — so weights
//! and activations are quantized to `uint8` with one scale and zero point
//! **per tensor**. That is the whole quantization vocabulary the device
//! has, and it is why a GGUF K-quant (a scale per 32-256 element block)
//! cannot be handed over unchanged.

use std::ffi::{CStr, CString, c_char, c_void};

use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};

/// Where the vendor BSP installs ONNX Runtime and the Zhouyi provider.
const DIR: &str = "/usr/share/cix/lib/onnxruntime";

/// The provider's compiler loads its operator library from here. It is read
/// from the environment, defaults to a relative `./operator`, and without it
/// session creation fails with "Cannot find layerlib under path".
const OPERATOR_PATH: &str = "OPERATOR_PATH";

/// Loaded before ONNX Runtime because the provider's libraries have no
/// `RUNPATH` and live in a directory that is not on the loader path.
/// Preloading them by absolute path with `RTLD_GLOBAL` satisfies those
/// references without requiring `LD_LIBRARY_PATH`.
const DEPS: &[&str] = &[
    "libaipu_driver.so",
    "libaipu_buildingtool.so",
    "libaipu_dsl.so",
    "libaipu_layerlib.so",
    "libcompass_runtime_midware.so",
    "libaiputoolchain_core.so",
    "libaiputoolchain.so",
];

/// Indices into `OrtApi`, which is a struct of function pointers in a fixed
/// order for a given API version.
///
/// Recovered from the shipped library rather than from a header: the table
/// was read at run time and each entry resolved against the binary's symbols.
/// They are checked again at load — see [`NpuOrt::open`] — because a silently
/// shifted index would be a call through the wrong signature.
mod slot {
    pub const GET_ERROR_MESSAGE: usize = 2;
    pub const CREATE_ENV: usize = 3;
    pub const CREATE_SESSION_FROM_ARRAY: usize = 8;
    pub const CREATE_SESSION_OPTIONS: usize = 10;
    pub const RELEASE_SESSION: usize = 95;
    pub const RELEASE_SESSION_OPTIONS: usize = 100;
    pub const ADD_SESSION_CONFIG_ENTRY: usize = 130;
}

/// `ORT_API_VERSION` matching the 1.20.0 runtime the BSP ships.
const API_VERSION: u32 = 20;

#[repr(C)]
struct OrtApiBase {
    get_api: unsafe extern "C" fn(u32) -> *const c_void,
    get_version_string: unsafe extern "C" fn() -> *const c_char,
}

/// What went wrong. `Run`-time failures carry ONNX Runtime's own message.
#[derive(Debug, Clone)]
pub struct OrtError(pub String);

impl std::fmt::Display for OrtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for OrtError {}

/// One tensor's quantization: `real = scale * (q - zero_point)`, with a
/// single pair for the whole tensor.
#[derive(Clone, Copy, Debug)]
pub struct Quant {
    pub scale: f32,
    pub zero_point: u8,
}

/// **Clipping the weight grid does not help on this device, measured twice.**
///
/// The standing problem is that [`Quant::covering`] spans `min..max`, so one
/// outlier sets the `uint8` step for a whole matrix and the bulk is left
/// with a handful of the 256 levels. Narrowing the range and letting the
/// tail saturate is the textbook answer, and it was worth trying because the
/// file format hinted at it: `Q4_K` stores weights on a coarse per-sub-block
/// grid, and a `Q4_K_M` checkpoint measures *better* through this device
/// than the `Q8_0` build of the same model when that model is badly
/// conditioned (gemma-3-1B: 10.2% and 7.6%) and *worse* when it is well
/// conditioned (gemma-3-4B: 4.3% and 4.8%). That is the signature of
/// tail-clipping helping a starved bulk, and it predicted the gemma-3-4B
/// pair before they were measured.
///
/// The prediction held and the mechanism still did not. Nine candidate
/// ranges shrunk toward the mean, `covering`'s own first so a tie changes
/// nothing, scored two ways over fifteen checkpoints at the width each one
/// actually serves:
///
/// | scored by | better | worse |
/// |---|---|---|
/// | error in `W` | 2 | 3 |
/// | error in `W·x` | 6 | 8 |
///
/// Scoring on `W·x` is the defensible objective — the weights a clip
/// damages are the large ones and the large ones dominate the sums — and it
/// did move more models, in both directions. What rules it out is the size
/// of the failures, not their count: llama-3.2-3B went 10.3% to **63.1%**
/// and gemma-2-2B 8.5% to 36.9%, and three checkpoints that currently
/// qualify (gemma-4-E2B, gemma-3-4B, qwen2.5-coder-0.5B) were pushed past
/// the bar. A change that sextuples the error on a model it was supposed to
/// help is not tuned, it is wrong: the host score is computed in `f32`
/// against exact activations, while the device quantizes the activations
/// too and rescales through an output range calibrated on the *unclipped*
/// projection, so the thing being minimised is not the thing being paid.
///
/// Together with the row-group split (`npu_tool`'s
/// `row_groups_against_one_scale_per_matrix`, which found 16 groups worth
/// only 1.8x on weight error and short of the bar), that is two independent
/// attacks on the one-scale-per-matrix ceiling and two negative results.
/// The remaining lever is per-channel scales, which this provider rejects.
#[cfg(doc)]
pub fn weight_grid_clipping_does_not_help() {}

/// A weight matrix's extremes, as [`Quant::covering`] wants them.
fn w_min_max(values: &[f32]) -> (f32, f32) {
    values
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)))
}

impl Quant {
    /// Picks a scale and zero point covering `[min, max]` in `uint8`.
    ///
    /// Asymmetric, because weights and activations are rarely centred on
    /// zero and a symmetric scale would waste half the range. A degenerate
    /// range (all values equal) still yields a usable scale rather than a
    /// division by zero.
    pub fn covering(min: f32, max: f32) -> Self {
        let (lo, hi) = (min.min(0.0), max.max(0.0));
        let span = hi - lo;
        let scale = if span > 0.0 { span / 255.0 } else { 1.0 };
        let zero_point = (-lo / scale).round().clamp(0.0, 255.0) as u8;
        Self { scale, zero_point }
    }

    pub fn quantize(&self, v: f32) -> u8 {
        ((v / self.scale).round() + self.zero_point as f32).clamp(0.0, 255.0) as u8
    }

    pub fn dequantize(&self, q: u8) -> f32 {
        (q as f32 - self.zero_point as f32) * self.scale
    }

    /// The interval this quantization can represent at all — level `0` and
    /// level `255` in `f32`.
    ///
    /// Not the same as the interval it was *calibrated* on, which sits
    /// inside this one by [`ACTIVATION_MARGIN`]. [`row_gain`] wants the
    /// representable one, because that is what a value has to stay inside
    /// to avoid being clamped.
    pub fn range(&self) -> (f32, f32) {
        let zero = f32::from(self.zero_point);
        (-zero * self.scale, (255.0 - zero) * self.scale)
    }
}

// ---------------------------------------------------------------------
// A minimal ONNX serializer.
//
// Only what a single quantized 1x1 convolution needs. Writing the protobuf
// directly keeps this dependency-free: pulling in a full ONNX crate to emit
// four nodes would be a large dependency for a very small message.
// ---------------------------------------------------------------------

fn varint(mut n: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (n & 0x7f) as u8;
        n >>= 7;
        if n == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn tag(field: u32, wire: u32, out: &mut Vec<u8>) {
    varint(((field << 3) | wire) as u64, out);
}

/// A length-delimited field: tag, byte length, then the payload.
fn delimited(field: u32, payload: &[u8], out: &mut Vec<u8>) {
    tag(field, 2, out);
    varint(payload.len() as u64, out);
    out.extend_from_slice(payload);
}

fn int_field(field: u32, value: u64, out: &mut Vec<u8>) {
    tag(field, 0, out);
    varint(value, out);
}

fn str_field(field: u32, value: &str, out: &mut Vec<u8>) {
    delimited(field, value.as_bytes(), out);
}

/// `TensorShapeProto`: a repeated `Dimension`, each holding a `dim_value`.
fn shape_proto(dims: &[i64]) -> Vec<u8> {
    let mut out = Vec::new();
    for d in dims {
        let mut dim = Vec::new();
        int_field(1, *d as u64, &mut dim);
        delimited(1, &dim, &mut out);
    }
    out
}

/// `TypeProto` wrapping a tensor of `elem_type` and shape.
fn type_proto(elem: u32, dims: &[i64]) -> Vec<u8> {
    let mut tensor = Vec::new();
    int_field(1, elem as u64, &mut tensor);
    delimited(2, &shape_proto(dims), &mut tensor);
    let mut out = Vec::new();
    delimited(1, &tensor, &mut out);
    out
}

fn value_info(name: &str, elem: u32, dims: &[i64]) -> Vec<u8> {
    let mut out = Vec::new();
    str_field(1, name, &mut out);
    delimited(2, &type_proto(elem, dims), &mut out);
    out
}

/// `TensorProto` for an initializer, carrying its bytes as `raw_data`.
fn initializer(name: &str, elem: u32, dims: &[i64], raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for d in dims {
        int_field(1, *d as u64, &mut out);
    }
    int_field(2, elem as u64, &mut out);
    str_field(8, name, &mut out);
    delimited(9, raw, &mut out);
    out
}

/// `AttributeProto` holding a list of ints (`AttributeType::INTS` = 7).
fn attr_ints(name: &str, values: &[i64]) -> Vec<u8> {
    let mut out = Vec::new();
    str_field(1, name, &mut out);
    for v in values {
        int_field(8, *v as u64, &mut out);
    }
    int_field(20, 7, &mut out);
    out
}

/// `AttributeProto` holding one int (`AttributeType::INT` = 2).
fn attr_int(name: &str, value: i64) -> Vec<u8> {
    let mut out = Vec::new();
    str_field(1, name, &mut out);
    int_field(3, value as u64, &mut out);
    int_field(20, 2, &mut out);
    out
}

fn node(inputs: &[&str], outputs: &[&str], name: &str, op: &str, attrs: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for i in inputs {
        str_field(1, i, &mut out);
    }
    for o in outputs {
        str_field(2, o, &mut out);
    }
    str_field(3, name, &mut out);
    str_field(4, op, &mut out);
    for a in attrs {
        delimited(5, a, &mut out);
    }
    out
}

const FLOAT: u32 = 1;
const UINT8: u32 = 2;

/// Builds the ONNX model for one quantized linear projection, expressed as
/// a 1x1 convolution.
///
/// Shapes follow the convolution view of a linear layer: the activation
/// arrives as `[1, k, m, 1]` — one image, `k` channels, `m` "rows", one
/// column — the weights as `[n, k, 1, 1]`, and the result is `[1, n, m, 1]`.
/// So the channel axis carries the input features and the height axis
/// carries the tokens.
///
/// The graph is `DequantizeLinear` on each of activation and weights, the
/// convolution in float, then `QuantizeLinear` on the result. That is the
/// standard QDQ form, and it is what lets the provider recognize the whole
/// thing as one quantized operation.
/// Builds a QDQ graph an operation at a time.
///
/// Every operation on this device is quantized, and the provider wants each
/// one wrapped in its own `DequantizeLinear` / `QuantizeLinear` pair — a "QDQ
/// node unit". Writing that by hand is three nodes and two initializers per
/// operation, which is where mistakes live. [`Self::op`] emits the whole unit
/// and hands back the name of its dequantized output, so a graph reads as the
/// sequence of operations it actually is.
struct QdqGraph {
    nodes: Vec<u8>,
    inits: Vec<u8>,
    next: usize,
}

impl QdqGraph {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inits: Vec::new(),
            next: 0,
        }
    }

    fn fresh(&mut self, prefix: &str) -> String {
        self.next += 1;
        format!("{prefix}{}", self.next)
    }

    /// Adds a scale/zero-point initializer pair, returning both names.
    fn quantization(&mut self, quant: Quant) -> (String, String) {
        let id = self.fresh("q");
        let (scale, zero) = (format!("{id}_scale"), format!("{id}_zp"));
        delimited(
            5,
            &initializer(&scale, FLOAT, &[], &quant.scale.to_le_bytes()),
            &mut self.inits,
        );
        delimited(
            5,
            &initializer(&zero, UINT8, &[], &[quant.zero_point]),
            &mut self.inits,
        );
        (scale, zero)
    }

    /// Adds a `uint8` initializer and dequantizes it, returning the float
    /// tensor's name. This is how a weight matrix enters a graph.
    fn constant(&mut self, values: &[u8], dims: &[i64], quant: Quant) -> String {
        let name = self.fresh("W");
        delimited(5, &initializer(&name, UINT8, dims, values), &mut self.inits);
        self.dequantize(&name, quant)
    }

    /// Dequantizes an existing `uint8` tensor.
    fn dequantize(&mut self, input: &str, quant: Quant) -> String {
        let (scale, zero) = self.quantization(quant);
        let out = self.fresh("f");
        let id = self.fresh("dq");
        delimited(
            1,
            &node(
                &[input, &scale, &zero],
                &[&out],
                &id,
                "DequantizeLinear",
                &[],
            ),
            &mut self.nodes,
        );
        out
    }

    /// Quantizes a float tensor into a named `uint8` tensor.
    fn quantize_into(&mut self, input: &str, output: &str, quant: Quant) {
        let (scale, zero) = self.quantization(quant);
        let id = self.fresh("q");
        delimited(
            1,
            &node(
                &[input, &scale, &zero],
                &[output],
                &id,
                "QuantizeLinear",
                &[],
            ),
            &mut self.nodes,
        );
    }

    /// Emits one operation as a complete QDQ unit: the operation, a
    /// `QuantizeLinear` of its result, and a `DequantizeLinear` back to
    /// float for whatever consumes it. Returns that float tensor's name.
    fn op(&mut self, op_type: &str, inputs: &[&str], attrs: &[Vec<u8>], quant: Quant) -> String {
        let quantized = self.op_quantized(op_type, inputs, attrs, quant);
        self.dequantize(&quantized, quant)
    }

    /// [`Self::op`], stopping at the `uint8` result.
    ///
    /// For a result that is read again at a different point in the graph, or
    /// that is the graph's own output. Note what this is *not* for: reading
    /// one tensor at two different quantizations. That looks like a way to
    /// fold a constant factor in for free, and it compiles, but a QDQ tensor
    /// carries one quantization and the provider keeps only one — the graph
    /// then computes something else entirely.
    fn op_quantized(
        &mut self,
        op_type: &str,
        inputs: &[&str],
        attrs: &[Vec<u8>],
        quant: Quant,
    ) -> String {
        let raw = self.fresh("t");
        let id = self.fresh(op_type);
        delimited(
            1,
            &node(inputs, &[&raw], &id, op_type, attrs),
            &mut self.nodes,
        );
        let quantized = self.fresh("u");
        self.quantize_into(&raw, &quantized, quant);
        quantized
    }

    /// Emits an operation whose quantized result *is* the graph's output, so
    /// it is not dequantized again.
    fn op_into(
        &mut self,
        op_type: &str,
        inputs: &[&str],
        attrs: &[Vec<u8>],
        output: &str,
        quant: Quant,
    ) {
        let raw = self.fresh("t");
        let id = self.fresh(op_type);
        delimited(
            1,
            &node(inputs, &[&raw], &id, op_type, attrs),
            &mut self.nodes,
        );
        self.quantize_into(&raw, output, quant);
    }

    /// Wraps the accumulated nodes and initializers into a complete model.
    fn finish(self, input: (&str, &[i64]), output: (&str, &[i64])) -> Vec<u8> {
        self.finish_inputs(&[input], output)
    }

    /// [`Self::finish`] with more than one graph input.
    ///
    /// Two inputs carrying the *same* bytes is how this graph breaks a
    /// dependency the provider otherwise miscompiles — see
    /// [`GateActivation::GeluSigmoidTwoInput`]. Duplicating a node does not
    /// work, because the provider merges the copies back together; two
    /// declared inputs are distinct tensors it cannot merge.
    fn finish_inputs(self, inputs: &[(&str, &[i64])], output: (&str, &[i64])) -> Vec<u8> {
        let mut graph = self.nodes;
        str_field(2, "orangu", &mut graph);
        graph.extend_from_slice(&self.inits);
        for input in inputs {
            delimited(11, &value_info(input.0, UINT8, input.1), &mut graph);
        }
        delimited(12, &value_info(output.0, UINT8, output.1), &mut graph);

        let mut model = Vec::new();
        int_field(1, 8, &mut model); // ir_version
        str_field(2, "orangu", &mut model);
        let mut opset = Vec::new();
        str_field(1, "", &mut opset);
        int_field(2, 13, &mut opset);
        delimited(8, &opset, &mut model);
        delimited(7, &graph, &mut model);
        model
    }
}

/// The 1x1 convolution attributes every projection here shares.
fn projection_attrs() -> Vec<Vec<u8>> {
    vec![
        attr_ints("kernel_shape", &[1, 1]),
        attr_ints("strides", &[1, 1]),
        attr_ints("pads", &[0, 0, 0, 0]),
        attr_ints("dilations", &[1, 1]),
        attr_int("group", 1),
    ]
}

/// Emits an ONNX model for a chain of linear projections, rectified between
/// them, quantized in the QDQ form the provider expects.
///
/// `dims` is `[in, hidden.., out]` and `weights[i]` is layer `i`'s already
/// quantized `[dims[i+1]][dims[i]]` filter. `acts` carries one quantization
/// per activation tensor — input, each intermediate, output — and `ws` one
/// per weight matrix.
///
/// Chaining matters more than it looks. A single projection spends more time
/// converting at the boundary than computing: 128x512x512 measures 0.29 ms
/// on the device against 0.69 ms including quantize and dequantize. Every
/// layer folded into one graph is a boundary crossing that does not happen,
/// because the intermediate stays on the device as `uint8` — worth a
/// measured 2.75x over three layers, and roughly half the artifact bytes,
/// since each separate graph carries its own input and output scaffolding.
fn build_stack_model(
    dims: &[usize],
    m: usize,
    weights: &[&[u8]],
    acts: &[Quant],
    ws: &[Quant],
) -> Vec<u8> {
    debug_assert_eq!(dims.len(), weights.len() + 1);
    debug_assert_eq!(acts.len(), weights.len() + 1);
    debug_assert_eq!(ws.len(), weights.len());

    let f32_bytes = |v: f32| v.to_le_bytes().to_vec();
    let layers = weights.len();

    // The input and output keep fixed names; everything between is numbered.
    let act_name = |i: usize| -> String {
        match i {
            0 => "X_q".into(),
            i if i == layers => "Y_q".into(),
            i => format!("T{i}_q"),
        }
    };

    let mut inits = Vec::new();
    for (i, quant) in acts.iter().enumerate() {
        delimited(
            5,
            &initializer(&format!("a{i}_scale"), FLOAT, &[], &f32_bytes(quant.scale)),
            &mut inits,
        );
        delimited(
            5,
            &initializer(&format!("a{i}_zp"), UINT8, &[], &[quant.zero_point]),
            &mut inits,
        );
    }
    for (i, quant) in ws.iter().enumerate() {
        // **One scale per output row**, not one per matrix. A single scale
        // spans the widest weight anywhere in the tensor, so a matrix with a
        // fat tail leaves its bulk far too few of the 256 levels — measured
        // as the step in standard deviations of `blk.0`'s gate, that is
        // 0.097 on gemma 4 E4B against 0.221 on llama 3.2 3B, and the block
        // error tracks it (2.6% against 12.1%). Per-row, a row's tail only
        // starves its own row.
        //
        // `axis = 0` is the output-channel axis of an `[n, k, 1, 1]` filter,
        // the one the vendor's toolchain says it supports (`just support
        // perchannel quantization in dimension 0`).
        delimited(
            5,
            &initializer(&format!("w{i}_scale"), FLOAT, &[], &f32_bytes(quant.scale)),
            &mut inits,
        );
        delimited(
            5,
            &initializer(&format!("w{i}_zp"), UINT8, &[], &[quant.zero_point]),
            &mut inits,
        );
        delimited(
            5,
            &initializer(
                &format!("W{i}_q"),
                UINT8,
                &[dims[i + 1] as i64, dims[i] as i64, 1, 1],
                weights[i],
            ),
            &mut inits,
        );
    }

    let conv_attrs = [
        attr_ints("kernel_shape", &[1, 1]),
        attr_ints("strides", &[1, 1]),
        attr_ints("pads", &[0, 0, 0, 0]),
        attr_ints("dilations", &[1, 1]),
        attr_int("group", 1),
    ];

    let mut nodes = Vec::new();
    for i in 0..layers {
        let (input, output) = (act_name(i), act_name(i + 1));
        let (x_f, w_f) = (format!("A{i}_f"), format!("W{i}_f"));
        let conv_out = format!("C{i}_f");

        delimited(
            1,
            &node(
                &[&input, &format!("a{i}_scale"), &format!("a{i}_zp")],
                &[&x_f],
                &format!("dqa{i}"),
                "DequantizeLinear",
                &[],
            ),
            &mut nodes,
        );
        delimited(
            1,
            &node(
                &[
                    &format!("W{i}_q"),
                    &format!("w{i}_scale"),
                    &format!("w{i}_zp"),
                ],
                &[&w_f],
                &format!("dqw{i}"),
                "DequantizeLinear",
                &[],
            ),
            &mut nodes,
        );
        delimited(
            1,
            &node(
                &[&x_f, &w_f],
                &[&conv_out],
                &format!("linear{i}"),
                "Conv",
                &conv_attrs,
            ),
            &mut nodes,
        );

        // No `Relu` node, and none is needed. A hidden layer's output is
        // quantized over `[0, bound]`, which puts its zero point at zero, and
        // `QuantizeLinear` into `uint8` clamps to `[0, 255]` — so every
        // negative value becomes zero on the way out. The rectifier is the
        // quantization.
        //
        // That is also the only form this stack compiles in. An explicit
        // `Relu` between the convolution and its `QuantizeLinear` makes the
        // provider reject the *convolution*: "Conv node `linear0` is not
        // supported" from `conv_op_builder.cc:352`, because its QDQ node
        // group no longer ends where the builder expects. Folding the
        // activation into the quantization sidesteps that and costs an
        // operation rather than adding one.
        delimited(
            1,
            &node(
                &[
                    &conv_out,
                    &format!("a{}_scale", i + 1),
                    &format!("a{}_zp", i + 1),
                ],
                &[&output],
                &format!("qa{}", i + 1),
                "QuantizeLinear",
                &[],
            ),
            &mut nodes,
        );
    }

    let mut graph = nodes;
    str_field(2, "linear", &mut graph);
    graph.extend_from_slice(&inits);
    delimited(
        11,
        &value_info("X_q", UINT8, &[1, dims[0] as i64, m as i64, 1]),
        &mut graph,
    );
    delimited(
        12,
        &value_info("Y_q", UINT8, &[1, dims[layers] as i64, m as i64, 1]),
        &mut graph,
    );

    let mut model = Vec::new();
    int_field(1, 8, &mut model); // ir_version
    str_field(2, "orangu", &mut model);
    let mut opset = Vec::new();
    str_field(1, "", &mut opset);
    int_field(2, 13, &mut opset);
    delimited(8, &opset, &mut model);
    delimited(7, &graph, &mut model);
    model
}

type FnGetErrorMessage = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type FnCreateEnv = unsafe extern "C" fn(i32, *const c_char, *mut *mut c_void) -> *mut c_void;
type FnCreateSessionFromArray = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    usize,
    *const c_void,
    *mut *mut c_void,
) -> *mut c_void;
type FnCreateSessionOptions = unsafe extern "C" fn(*mut *mut c_void) -> *mut c_void;
type FnAddSessionConfigEntry =
    unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> *mut c_void;
type FnRelease = unsafe extern "C" fn(*mut c_void);

/// The ONNX Runtime entry points this module calls.
///
/// Short, because this module only compiles. Nothing here runs a model —
/// see the note on one inference per session.
struct Api {
    get_error_message: FnGetErrorMessage,
    create_env: FnCreateEnv,
    create_session_from_array: FnCreateSessionFromArray,
    create_session_options: FnCreateSessionOptions,
    add_session_config_entry: FnAddSessionConfigEntry,
    release_session: FnRelease,
    release_session_options: FnRelease,
}

/// The resolved API table and the ONNX Runtime environment.
///
/// `Rc` rather than `Arc` because nothing in this stack is documented
/// thread-safe.
struct OrtInner {
    api: Api,
    env: *mut c_void,
}

/// An open ONNX Runtime with the Zhouyi provider available.
pub struct NpuOrt {
    inner: std::rc::Rc<OrtInner>,
    append_zhouyi: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
}

/// The shape of one projection, gathered rather than passed one argument at
/// a time — `k`, `n` and `max_tokens` are three facts about the same matrix
/// and travel together everywhere they go.
#[derive(Clone, Copy)]
struct ProjectionShape {
    k: usize,
    n: usize,
    max_tokens: usize,
}

impl NpuOrt {
    /// Opens ONNX Runtime, or `None` when the vendor stack is not installed
    /// — the same "absence is an answer" contract [`crate::npu`] has.
    pub fn open() -> Option<Self> {
        // SAFETY: the provider's libraries carry no RUNPATH and sit outside
        // the loader path, so they are preloaded by absolute path with
        // RTLD_GLOBAL to satisfy ONNX Runtime's references to them.
        let mut library_deps = Vec::new();
        for dep in DEPS {
            if let Ok(lib) =
                unsafe { Library::open(Some(&format!("{DIR}/{dep}")), RTLD_NOW | RTLD_GLOBAL) }
            {
                library_deps.push(lib);
            }
        }
        // SAFETY: as above.
        let library = unsafe {
            Library::open(
                Some(&format!("{DIR}/libonnxruntime.so.1.20.0")),
                RTLD_NOW | RTLD_GLOBAL,
            )
        }
        .ok()?;

        // The provider's compiler reads this from the environment when a
        // session is created, and fails without it. Set only when the
        // operator has not chosen a path themselves.
        //
        // SAFETY: `set_var` is not thread-safe, so this must happen during
        // initialization, before a session exists and before any other
        // thread reads the environment. `open` is that moment.
        if std::env::var_os(OPERATOR_PATH).is_none() {
            unsafe { std::env::set_var(OPERATOR_PATH, format!("{DIR}/operator")) };
        }

        // SAFETY: `OrtGetApiBase` is ONNX Runtime's documented entry point
        // and takes no arguments.
        let api = unsafe {
            let base: libloading::os::unix::Symbol<unsafe extern "C" fn() -> *const OrtApiBase> =
                library.get(b"OrtGetApiBase\0").ok()?;
            let base = base();
            if base.is_null() {
                return None;
            }
            let table = ((*base).get_api)(API_VERSION) as *const *const c_void;
            if table.is_null() {
                return None;
            }
            // Every slot used below must be populated; a null one would mean
            // the table is not the shape this module was built against.
            for index in [
                slot::GET_ERROR_MESSAGE,
                slot::CREATE_ENV,
                slot::CREATE_SESSION_FROM_ARRAY,
                slot::CREATE_SESSION_OPTIONS,
                slot::ADD_SESSION_CONFIG_ENTRY,
                slot::RELEASE_SESSION,
                slot::RELEASE_SESSION_OPTIONS,
            ] {
                if (*table.add(index)).is_null() {
                    return None;
                }
            }
            let at = |i: usize| *table.add(i);
            Api {
                get_error_message: std::mem::transmute::<*const c_void, FnGetErrorMessage>(at(
                    slot::GET_ERROR_MESSAGE,
                )),
                create_env: std::mem::transmute::<*const c_void, FnCreateEnv>(at(slot::CREATE_ENV)),
                create_session_from_array: std::mem::transmute::<
                    *const c_void,
                    FnCreateSessionFromArray,
                >(at(slot::CREATE_SESSION_FROM_ARRAY)),
                create_session_options: std::mem::transmute::<*const c_void, FnCreateSessionOptions>(
                    at(slot::CREATE_SESSION_OPTIONS),
                ),
                add_session_config_entry: std::mem::transmute::<
                    *const c_void,
                    FnAddSessionConfigEntry,
                >(at(slot::ADD_SESSION_CONFIG_ENTRY)),
                release_session: std::mem::transmute::<*const c_void, FnRelease>(at(
                    slot::RELEASE_SESSION,
                )),
                release_session_options: std::mem::transmute::<*const c_void, FnRelease>(at(
                    slot::RELEASE_SESSION_OPTIONS,
                )),
            }
        };

        // The vendor libraries are deliberately never unloaded. They carry
        // global state — operator registries, a device singleton, static
        // maps built on first use — and running their unload path partway
        // through a process crashes it: the test suite segfaulted at exit
        // until these were leaked. Since the API table below points into
        // this code, keeping it mapped for the life of the process is also
        // what makes those pointers valid.
        std::mem::forget(library_deps);

        // SAFETY: the provider is appended through its own exported C symbol
        // rather than through the API table.
        let append_zhouyi = unsafe {
            let s: libloading::os::unix::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
                library
                    .get(b"OrtSessionOptionsAppendExecutionProvider_Zhouyi\0")
                    .ok()?;
            *s
        };
        std::mem::forget(library);

        let tag = CString::new("orangu").ok()?;
        let mut env: *mut c_void = std::ptr::null_mut();
        // SAFETY: single out-pointer; 2 is ORT_LOGGING_LEVEL_WARNING.
        unsafe {
            if !(api.create_env)(2, tag.as_ptr(), &mut env).is_null() || env.is_null() {
                return None;
            }
        }

        Some(Self {
            inner: std::rc::Rc::new(OrtInner { api, env }),
            append_zhouyi,
        })
    }

    /// Turns an ONNX Runtime status into an error carrying its message.
    ///
    /// # Safety
    /// `status` must be a non-null status returned by ONNX Runtime.
    unsafe fn message(&self, status: *mut c_void, what: &str) -> OrtError {
        // SAFETY: the caller guarantees `status` came from ONNX Runtime and
        // is non-null, so it owns a NUL-terminated message.
        let msg = unsafe { CStr::from_ptr((self.inner.api.get_error_message)(status)) }
            .to_string_lossy()
            .into_owned();
        OrtError(format!("{what}: {msg}"))
    }

    /// Compiles one linear projection for the NPU.
    ///
    /// `weights` is row-major `[n][k]` — output feature major, matching the
    /// convolution's `[n, k, 1, 1]` filter layout. `x_range` is the expected
    /// activation range, which static quantization needs up front: the scale
    /// is baked into the compiled graph, so a value outside it saturates.
    ///
    /// `max_tokens` fixes the token axis. The compiled graph has a static
    /// shape, so a different token count needs a different graph.
    ///
    /// This is the expensive call — a few hundred milliseconds. The result
    /// is worth keeping: it can be bound to a runtime as many times as you
    /// like, and its `binary` can be written to disk and compiled on another
    /// machine entirely.
    pub fn compile_linear(
        &self,
        weights: &[f32],
        k: usize,
        n: usize,
        max_tokens: usize,
        x_range: (f32, f32),
    ) -> Result<CompiledLinear, OrtError> {
        if weights.len() != k * n {
            return Err(OrtError(format!(
                "weights are {} values, expected k*n = {}",
                weights.len(),
                k * n
            )));
        }
        if k == 0 || n == 0 || max_tokens == 0 {
            return Err(OrtError("a dimension is zero".into()));
        }
        let x_quant = Quant::covering(x_range.0, x_range.1);
        let bound = worst_case_magnitude(weights, k, n, x_range);
        let y_quant = Quant::covering(-bound, bound);

        let (w_min, w_max) = w_min_max(weights);
        let w_quant = Quant::covering(w_min, w_max);
        let w_q: Vec<u8> = weights.iter().map(|v| w_quant.quantize(*v)).collect();
        let model = build_stack_model(
            &[k, n],
            max_tokens,
            &[&w_q],
            &[x_quant, y_quant],
            &[w_quant],
        );

        let context = ContextFile::new()?;
        let binary = self.compile(&model, &context)?;

        Ok(CompiledLinear {
            binary,
            k,
            n,
            max_tokens,
            inputs: 1,
            x: x_quant,
            y: y_quant,
            unsmoothing: Vec::new(),
        })
    }

    /// One projection at chosen quantizations.
    ///
    /// [`Self::compile_linear`] derives a worst-case output range because a
    /// lone projection can afford one. A caller that has calibration — and
    /// so knows the range the output actually occupies — passes it here
    /// instead and gets a far finer output step.
    /// Per-input-channel scales that move range out of the activations and
    /// into the weights, where a per-tensor scale can carry it.
    ///
    /// **The problem this exists for**, measured on a real gemma 4 E2B
    /// hidden state: the largest value in a row is 38.9 times that row's
    /// RMS, and it is a handful of channels that put it there. Both
    /// quantizers here are per *tensor*, so those few channels set the step
    /// size for all 2560 of them, and everything else is left with a
    /// fraction of the 256 levels it could have had.
    ///
    /// `x . W` does not care how the magnitude is split between the two, so
    /// it can be moved: divide channel `j` of the input by `s_j` and
    /// multiply column `j` of the weights by `s_j` and the product is
    /// identical. Choosing `s_j = max|x_j|^a / max|W_j|^(1-a)` leaves both
    /// sides at `(max|x_j| max|W_j|)^a`-ish — flat across channels rather
    /// than dominated by a few — which is the whole trick, and it is
    /// SmoothQuant's.
    ///
    /// `a = 0.5` splits the difference evenly. A channel with no signal on
    /// either side gets 1.0 rather than a division by zero: it contributes
    /// nothing to the product, so any scale is correct and this is the one
    /// that cannot overflow.
    ///
    /// **What it is worth**, on gemma 4 E2B at `Q8_0` calibrated on its own
    /// captured activations, measured against an `f32` reference on real
    /// input:
    ///
    /// | block | one scale per tensor | smoothed |
    /// |---|---|---|
    /// | `blk.0` | 20.0% rms | **5.2%** |
    /// | `blk.5` | 22.8% | **7.1%** |
    /// | `blk.16` | 18.3% | **10.6%** |
    ///
    /// Two to four times less error for one multiply per input element at
    /// run time, and nothing at all on the device — the weights carry the
    /// other half of the factor, so the graph is the same shape and the
    /// same speed.
    /// The migration strengths [`Self::choose_smoothing_alpha`] tries.
    ///
    /// `0.5` is the value SmoothQuant's paper uses and the only one this
    /// used to have. It is the right default and it is not right
    /// everywhere: a projection whose input is dominated by a few channels
    /// needs more of that outlier moved into the weights before the bulk
    /// gets any levels at all, and `down` — whose input is
    /// `silu(gate) * up`, a product of two things that are already peaked —
    /// is where that shows. On llama 3.2 3B `Q8_0`, layer 1's `down` input
    /// used **9 of 256 levels** at `0.5`, and a decode row through it came
    /// back exactly zero.
    const ALPHA_CANDIDATES: [f32; 5] = [0.5, 0.65, 0.75, 0.85, 0.95];

    /// How many output channels [`Self::choose_smoothing_alpha`] scores.
    ///
    /// The statistic is an average over output channels, so it converges
    /// long before every one of them is included, and the cost is linear in
    /// this: 64 channels against `d_ff`'s 8192 is a hundredth of the work
    /// for a number that picks the same `alpha`.
    const ALPHA_SAMPLE_CHANNELS: usize = 64;

    /// Picks the migration strength for one projection by simulating what
    /// it costs, rather than taking 0.5 because the paper does.
    ///
    /// The simulation is the real thing on the host: smooth the activations
    /// and the weights by `s(alpha)`, quantize both exactly as
    /// [`Self::compile_projection`] and [`observed_quant`] will, and
    /// measure how far the result lands from the `f32` answer. Cheap
    /// because it scores a sample of output channels — the error is an
    /// average over them — so it adds a fraction of a second to a compile
    /// that takes about a minute.
    ///
    /// Ties go to the smallest `alpha` (the candidates are tried in order
    /// and only a strict improvement displaces the incumbent), so a
    /// projection that does not care keeps the value it always had.
    fn choose_smoothing_alpha(activations: &[f32], weights: &[f32], k: usize, n: usize) -> f32 {
        let rows = activations.len() / k.max(1);
        if rows == 0 || n == 0 || weights.len() < n * k {
            return Self::ALPHA_CANDIDATES[0];
        }
        // Per-input-channel signed extremes of the weights, once: the
        // smoothed matrix's global range is these scaled by `s`, since
        // every `s` is positive.
        let (mut col_lo, mut col_hi) = (vec![0.0f32; k], vec![0.0f32; k]);
        for row in weights.chunks_exact(k) {
            for ((lo, hi), w) in col_lo.iter_mut().zip(&mut col_hi).zip(row) {
                *lo = lo.min(*w);
                *hi = hi.max(*w);
            }
        }
        let channels: Vec<usize> = sampled_channels(n, Self::ALPHA_SAMPLE_CHANNELS);
        // The `f32` answer to compare against, once — it does not depend on
        // `alpha`.
        let mut exact = vec![0.0f32; rows * channels.len()];
        for (t, x) in activations.chunks_exact(k).enumerate() {
            for (c, &out) in channels.iter().enumerate() {
                let w = &weights[out * k..out * k + k];
                exact[t * channels.len() + c] = x.iter().zip(w).map(|(a, b)| a * b).sum::<f32>();
            }
        }

        let mut best = (Self::ALPHA_CANDIDATES[0], f32::INFINITY);
        let mut smoothed_x = vec![0.0f32; activations.len()];
        for alpha in Self::ALPHA_CANDIDATES {
            let s = Self::smoothing_scales(activations, weights, k, alpha);
            for (row, out) in activations
                .chunks_exact(k)
                .zip(smoothed_x.chunks_exact_mut(k))
            {
                for ((v, s), o) in row.iter().zip(&s).zip(out) {
                    *o = v / s;
                }
            }
            let x_q = observed_quant(&smoothed_x);
            let (w_lo, w_hi) = col_lo
                .iter()
                .zip(&col_hi)
                .zip(&s)
                .fold((0.0f32, 0.0f32), |(lo, hi), ((l, h), s)| {
                    (lo.min(l * s), hi.max(h * s))
                });
            let w_q = Quant::covering(w_lo, w_hi);

            let (mut num, mut den) = (0.0f64, 0.0f64);
            for (t, x) in smoothed_x.chunks_exact(k).enumerate() {
                for (c, &out) in channels.iter().enumerate() {
                    let w = &weights[out * k..out * k + k];
                    let got: f32 = x
                        .iter()
                        .zip(w)
                        .zip(&s)
                        .map(|((a, b), s)| {
                            x_q.dequantize(x_q.quantize(*a)) * w_q.dequantize(w_q.quantize(b * s))
                        })
                        .sum();
                    let want = exact[t * channels.len() + c];
                    num += f64::from(got - want).powi(2);
                    den += f64::from(want).powi(2);
                }
            }
            let error = if den > 0.0 {
                (num / den).sqrt() as f32
            } else {
                0.0
            };
            if error < best.1 {
                best = (alpha, error);
            }
        }
        best.0
    }

    fn smoothing_scales(activations: &[f32], weights: &[f32], k: usize, alpha: f32) -> Vec<f32> {
        let mut x_max = vec![0.0f32; k];
        for row in activations.chunks_exact(k) {
            for (m, v) in x_max.iter_mut().zip(row) {
                *m = m.max(v.abs());
            }
        }
        let mut w_max = vec![0.0f32; k];
        for row in weights.chunks_exact(k) {
            for (m, v) in w_max.iter_mut().zip(row) {
                *m = m.max(v.abs());
            }
        }
        x_max
            .iter()
            .zip(&w_max)
            .map(|(x, w)| {
                if *x <= 0.0 || *w <= 0.0 || !x.is_finite() || !w.is_finite() {
                    return 1.0;
                }
                let s = x.powf(alpha) / w.powf(1.0 - alpha);
                if s.is_finite() && s > 0.0 { s } else { 1.0 }
            })
            .collect()
    }

    fn compile_projection(
        &self,
        weights: &[f32],
        shape: ProjectionShape,
        x_quant: Quant,
        y_quant: Quant,
        smoothing: &[f32],
    ) -> Result<CompiledLinear, OrtError> {
        let ProjectionShape { k, n, max_tokens } = shape;
        // **The smoothing is folded into the weights here.** `x . W` is
        // unchanged by dividing input channel `j` by `s_j` and multiplying
        // `W[:, j]` by the same, so this costs one multiply per input
        // element at run time and nothing at all in accuracy — see
        // [`smoothing_scales`] for why it is worth doing.
        let smoothed: Vec<f32>;
        let weights = if smoothing.is_empty() {
            weights
        } else {
            smoothed = weights
                .chunks_exact(k)
                .flat_map(|row| row.iter().zip(smoothing).map(|(w, s)| w * s))
                .collect();
            &smoothed
        };

        // Plain `covering`, and *not* for want of trying something
        // cleverer — see `weight_grid_clipping_does_not_help` for the two
        // measured attempts and what they cost.
        let (w_min, w_max) = w_min_max(weights);
        let w_quant = Quant::covering(w_min, w_max);
        let w_q: Vec<u8> = weights.iter().map(|v| w_quant.quantize(*v)).collect();
        let model = build_stack_model(
            &[k, n],
            max_tokens,
            &[&w_q],
            &[x_quant, y_quant],
            &[w_quant],
        );
        let context = ContextFile::new()?;
        let binary = self.compile(&model, &context)?;
        Ok(CompiledLinear {
            binary,
            k,
            n,
            max_tokens,
            inputs: 1,
            x: x_quant,
            y: y_quant,
            // The reciprocal, so the serving path multiplies rather than
            // divides once per input element.
            unsmoothing: smoothing.iter().map(|s| 1.0 / s).collect(),
        })
    }

    /// A gated feed-forward block as **three projections**, with the
    /// activation computed on the host.
    ///
    /// The alternative — the whole block as one graph
    /// ([`Self::compile_gated_ffn`]) — is faster on paper and wrong on this
    /// provider. That function's doc comment records the bisection: every
    /// component is correct alone, and `gate * f(gate)` for a nonlinear `f`
    /// comes back about 50% wrong no matter how the subgraph is spelled,
    /// including with the two multiplicands built from two separate graph
    /// inputs so nothing can be merged. Three spellings produced
    /// bit-identical wrong answers, one of them while measurably executing an
    /// extra convolution, so it is not something a graph rewrite reaches.
    ///
    /// So the activation does not go to the device at all. The projections
    /// do, one graph each, and each of those is a shape this device is known
    /// to compute correctly — `compile_linear`'s own cross-check covers it,
    /// and a 10240-wide `ffn_down` measured inside the device's quantization
    /// step.
    ///
    /// The costs are real: three job submissions per block instead of one,
    /// the `d_ff`-wide gate and up activations crossing the host boundary in
    /// both directions, and `gelu(gate) * up` on the CPU. Measured on
    /// `blk.0` of Gemma 4 E4B (2560 x 10240) at 128 tokens:
    ///
    /// | | time | rate | error vs `f32` |
    /// |---|---|---|---|
    /// | one graph ([`Self::compile_gated_ffn`]) | 19.1 ms | 1054 GF/s | 39% — wrong |
    /// | three graphs, this | **45.4 ms** | 443 GF/s | **3.5%** |
    /// | `CpuBackend`, whole block | 92.5 ms | 218 GF/s | — |
    ///
    /// So the split costs 26 ms of the one-graph form's speed and buys a
    /// correct answer, and it is still **2.0x** the CPU. The 3.5% is the
    /// per-tensor `uint8` quantization itself and nothing else: the host
    /// simulation of this exact pipeline
    /// (`_scratch_simulated_gated_ffn_error`) predicted 3.9% for the same
    /// block, which is the number this now lands on — the 39% really was all
    /// the device's activation.
    ///
    /// Decode is a different question and the answer there is no: at one
    /// token a whole block is 3.02 ms on the device against 2.85 ms on the
    /// CPU, because both read the same ~79 MB of weights from the same
    /// LPDDR and neither can beat the other at it. This is a prefill path.
    /// Compiles several projections that all read the **same** input, with
    /// nothing between them.
    ///
    /// `Q`, `K` and `V` are exactly this shape: three matrices against one
    /// `attn_norm(x)`, no activation, no chaining. It is the gated block
    /// minus the activation and minus `down`, so it reuses the same
    /// calibrated machinery — per-projection smoothing chosen by
    /// [`Self::choose_smoothing_alpha`], input and output ranges from
    /// [`observed_quant`] over the real activations — rather than
    /// [`Self::compile_linear`]'s derived worst-case range, which is
    /// pessimistic enough to be worth avoiding wherever a capture exists.
    ///
    /// `shapes` is each projection's output width, in order; every one
    /// takes `k` inputs. `calibration` is `[rows][k]` row-major.
    pub fn compile_shared_input(
        &self,
        weights: &[&[f32]],
        shapes: &[usize],
        k: usize,
        max_tokens: usize,
        calibration: &[f32],
    ) -> Result<Vec<CompiledLinear>, OrtError> {
        if weights.len() != shapes.len() {
            return Err(OrtError(format!(
                "{} weight matrices for {} shapes",
                weights.len(),
                shapes.len()
            )));
        }
        if k == 0 || max_tokens == 0 {
            return Err(OrtError("a dimension is zero".into()));
        }
        if calibration.is_empty() || !calibration.len().is_multiple_of(k) {
            return Err(OrtError(format!(
                "calibration is {} values, not a whole number of {k}-feature rows",
                calibration.len()
            )));
        }
        let rows = calibration.len() / k;
        let mut out = Vec::with_capacity(weights.len());
        for (w, &n) in weights.iter().zip(shapes) {
            if w.len() != n * k {
                return Err(OrtError(format!(
                    "a projection has {} weights, expected n*k = {}",
                    w.len(),
                    n * k
                )));
            }
            // The projection on the host in `f32`, so its output range is
            // the one it actually occupies — the same reasoning as
            // `compile_gated_ffn_host`.
            let mut y = vec![0.0f32; rows * n];
            for r in 0..rows {
                for o in 0..n {
                    let mut acc = 0.0f32;
                    for i in 0..k {
                        acc += calibration[r * k + i] * w[o * k + i];
                    }
                    y[r * n + o] = acc;
                }
            }
            let alpha = Self::choose_smoothing_alpha(calibration, w, k, n);
            let scales = Self::smoothing_scales(calibration, w, k, alpha);
            let smoothed: Vec<f32> = calibration
                .chunks_exact(k)
                .flat_map(|row| row.iter().zip(&scales).map(|(v, s)| v / s))
                .collect();
            out.push(self.compile_projection(
                w,
                ProjectionShape { k, n, max_tokens },
                observed_quant(&smoothed),
                observed_quant(&y),
                &scales,
            )?);
        }
        Ok(out)
    }

    pub fn compile_gated_ffn_host(
        &self,
        ffn: &GatedFfn<'_>,
        max_tokens: usize,
        calibration: &[f32],
        activation: HostActivation,
    ) -> Result<CompiledGatedFfn, OrtError> {
        let GatedFfn {
            d_model,
            d_ff,
            gate,
            up,
            down,
        } = *ffn;
        if d_model == 0 || d_ff == 0 || max_tokens == 0 {
            return Err(OrtError("a dimension is zero".into()));
        }
        for (name, weights, expected) in [
            ("gate", gate, d_ff * d_model),
            ("up", up, d_ff * d_model),
            ("down", down, d_model * d_ff),
        ] {
            if weights.len() != expected {
                return Err(OrtError(format!(
                    "{name} has {} weights, expected {expected}",
                    weights.len()
                )));
            }
        }
        if calibration.is_empty() || !calibration.len().is_multiple_of(d_model) {
            return Err(OrtError(format!(
                "calibration is {} values, not a whole number of {d_model}-feature rows",
                calibration.len()
            )));
        }

        // The block on the host in `f32`, so every tensor is quantized to the
        // range it actually occupies — the same reasoning as
        // [`Self::compile_stack`], and for the same reason.
        let rows = calibration.len() / d_model;
        let project = |x: &[f32], w: &[f32], in_dim: usize, out_dim: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; rows * out_dim];
            for r in 0..rows {
                for o in 0..out_dim {
                    let mut acc = 0.0f32;
                    for i in 0..in_dim {
                        acc += x[r * in_dim + i] * w[o * in_dim + i];
                    }
                    out[r * out_dim + o] = acc;
                }
            }
            out
        };
        let gate_out = project(calibration, gate, d_model, d_ff);
        let up_out = project(calibration, up, d_model, d_ff);
        let hidden: Vec<f32> = gate_out
            .iter()
            .zip(&up_out)
            .map(|(g, u)| activation.apply(*g) * u)
            .collect();
        let y = project(&hidden, down, d_ff, d_model);

        // One set of scales per projection, because each has its own input
        // and its own weights — `down` reads the hidden state, whose
        // outliers are not the input's. `hidden` is already computed above
        // for the output ranges, so this costs nothing extra.
        // One `alpha` per projection, chosen by measurement — see
        // [`Self::choose_smoothing_alpha`]. `down` is the one that usually
        // wants a different answer from the other two.
        let gate_s = Self::smoothing_scales(
            calibration,
            gate,
            d_model,
            Self::choose_smoothing_alpha(calibration, gate, d_model, d_ff),
        );
        let up_s = Self::smoothing_scales(
            calibration,
            up,
            d_model,
            Self::choose_smoothing_alpha(calibration, up, d_model, d_ff),
        );
        let down_s = Self::smoothing_scales(
            &hidden,
            down,
            d_ff,
            Self::choose_smoothing_alpha(&hidden, down, d_ff, d_model),
        );

        // The ranges have to be measured on what the graph will actually be
        // fed, which is the *smoothed* input — measuring the original and
        // then quantizing the other thing is the bug this whole area is
        // prone to.
        let smooth = |x: &[f32], s: &[f32], k: usize| -> Vec<f32> {
            x.chunks_exact(k)
                .flat_map(|row| row.iter().zip(s).map(|(v, s)| v / s))
                .collect()
        };
        let x_q = observed_quant(&smooth(calibration, &gate_s, d_model));
        let up_x_q = observed_quant(&smooth(calibration, &up_s, d_model));
        let hidden_q = observed_quant(&smooth(&hidden, &down_s, d_ff));
        Ok(CompiledGatedFfn {
            activation,
            gate: self.compile_projection(
                gate,
                ProjectionShape {
                    k: d_model,
                    n: d_ff,
                    max_tokens,
                },
                x_q,
                observed_quant(&gate_out),
                &gate_s,
            )?,
            up: self.compile_projection(
                up,
                ProjectionShape {
                    k: d_model,
                    n: d_ff,
                    max_tokens,
                },
                up_x_q,
                observed_quant(&up_out),
                &up_s,
            )?,
            down: self.compile_projection(
                down,
                ProjectionShape {
                    k: d_ff,
                    n: d_model,
                    max_tokens,
                },
                hidden_q,
                observed_quant(&y),
                &down_s,
            )?,
        })
    }

    /// Compiles a chain of rectified projections into **one** graph.
    ///
    /// `dims` is `[in, hidden.., out]`, and `weights[i]` is layer `i`'s
    /// row-major `[dims[i+1]][dims[i]]` matrix. `calibration` is
    /// representative input, row-major `[rows][dims[0]]` — see below.
    ///
    /// Prefer this to a projection at a time whenever the layers are
    /// consecutive. The result has the same interface as a single one — it
    /// takes `dims[0]` features and returns the last — but the intermediates
    /// never leave the device, so the quantize/dequantize cost is paid once
    /// for the whole chain rather than once per layer. On this device that
    /// boundary is over half the wall time of a single projection.
    ///
    /// # Why this needs calibration and [`Self::compile_linear`] does not
    ///
    /// A single layer can use a derived worst-case output range: pessimistic,
    /// but it only has to be survivable once. Chained, that reasoning
    /// collapses. Each layer's worst case becomes the next layer's assumed
    /// input, and the bound compounds — two layers of it put the real
    /// activations so far below full scale that everything quantized to zero
    /// and the graph returned nothing but zeros. Measured ranges are not an
    /// optimization here; they are what makes a stack compute anything.
    ///
    /// So the layers are run on the host over `calibration`, in `f32`, and
    /// each activation is quantized to the range actually observed, widened
    /// by a margin for inputs the calibration did not cover.
    pub fn compile_stack(
        &self,
        dims: &[usize],
        weights: &[&[f32]],
        max_tokens: usize,
        calibration: &[f32],
    ) -> Result<CompiledLinear, OrtError> {
        if weights.is_empty() || dims.len() != weights.len() + 1 {
            return Err(OrtError(format!(
                "{} weight matrices need {} dimensions, {} given",
                weights.len(),
                weights.len() + 1,
                dims.len()
            )));
        }
        if let Some(bad) = dims.iter().position(|d| *d == 0) {
            return Err(OrtError(format!("dimension {bad} is zero")));
        }
        if max_tokens == 0 {
            return Err(OrtError("max_tokens is zero".into()));
        }
        for (i, layer) in weights.iter().enumerate() {
            let expected = dims[i] * dims[i + 1];
            if layer.len() != expected {
                return Err(OrtError(format!(
                    "layer {i} has {} weights, expected {} x {} = {expected}",
                    layer.len(),
                    dims[i + 1],
                    dims[i],
                )));
            }
        }
        if calibration.is_empty() {
            return Err(OrtError("calibration input is empty".into()));
        }
        if !calibration.len().is_multiple_of(dims[0]) {
            return Err(OrtError(format!(
                "calibration is {} values, not a whole number of {}-feature rows",
                calibration.len(),
                dims[0]
            )));
        }

        let mut activations = calibration.to_vec();
        let mut acts = vec![observed_quant(&activations)];
        let mut ws = Vec::with_capacity(weights.len());
        let mut quantized = Vec::with_capacity(weights.len());

        for (i, layer) in weights.iter().enumerate() {
            let quant = {
                let (lo, hi) = layer
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
                Quant::covering(lo, hi)
            };
            quantized.push(
                layer
                    .iter()
                    .map(|v| quant.quantize(*v))
                    .collect::<Vec<u8>>(),
            );
            ws.push(quant);

            // The layer as the device will compute it, in `f32`: project,
            // then rectify unless this is the last one, matching the
            // `[0, bound]` quantization that carries the rectifier.
            let (in_features, out_features) = (dims[i], dims[i + 1]);
            let rows = activations.len() / in_features;
            let mut next = vec![0.0f32; rows * out_features];
            for r in 0..rows {
                for o in 0..out_features {
                    let mut acc = 0.0f32;
                    for f in 0..in_features {
                        acc += activations[r * in_features + f] * layer[o * in_features + f];
                    }
                    next[r * out_features + o] = if i + 1 < weights.len() {
                        acc.max(0.0)
                    } else {
                        acc
                    };
                }
            }
            activations = next;
            acts.push(observed_quant(&activations));
        }

        let borrowed: Vec<&[u8]> = quantized.iter().map(|w| w.as_slice()).collect();
        let model = build_stack_model(dims, max_tokens, &borrowed, &acts, &ws);

        let context = ContextFile::new()?;
        let binary = self.compile(&model, &context)?;

        Ok(CompiledLinear {
            binary,
            k: dims[0],
            n: dims[weights.len()],
            max_tokens,
            inputs: 1,
            x: acts[0],
            y: acts[weights.len()],
            unsmoothing: Vec::new(),
        })
    }

    /// Compiles a gated feed-forward block — `down(act(gate(x)) * up(x))` —
    /// into one graph.
    ///
    /// `calibration` is representative input, row-major `[rows][d_model]`.
    ///
    /// This is the block worth moving to the device. In a Gemma 4 layer the
    /// feed-forward network is the large majority of the arithmetic, and its
    /// three projections as three separate graphs pay the host boundary
    /// three times over — where fusing pays it once.
    /// A whole gated feed-forward block as one graph.
    ///
    /// **This is accurate on Gemma 4's vision blocks and not on its
    /// language-model blocks, and the cause is in the provider, not here.**
    /// Speed is not the problem: `blk.0` runs at 1054 GF/s against the CPU's
    /// 217 GF/s for the same 128-token block. It comes back ~39% wrong
    /// (6.30 against a magnitude of 16.06).
    ///
    /// The error was bisected by compiling the same block with the gate
    /// chain cut back a piece at a time, all at `768 x 3072` so shape is
    /// held fixed:
    ///
    /// | gate chain | on the device |
    /// |---|---|
    /// | [`GateActivation::None`] — no activation | passes |
    /// | [`GateActivation::ScaleOnly`] — `gate * k` | passes |
    /// | [`GateActivation::FanOut`] — `gate * (gate * k)` | passes |
    /// | [`GateActivation::GeluSigmoid`] — `gate * Sigmoid(gate * k)` | fails |
    /// | [`GateActivation::Gelu`] — `gate * (Erf(gate * c) + 1)` | fails |
    ///
    /// So the quantized convolutions are right, the scalar-constant encoding
    /// is right, and reading the dequantized gate twice is right. The error
    /// appears exactly when a `Sigmoid`, or an `Erf` and `Add`, is put in
    /// the chain — and the two failures are the same size, which two
    /// unrelated activations approximating the same function should not be
    /// unless what is wrong is common to both.
    ///
    /// Three explanations were measured and eliminated first, so they do not
    /// get proposed again. **Quantization**: `_scratch_simulated_gated_ffn_error`
    /// replays this exact pipeline in `f32` — every stage rounded through
    /// its own [`observed_quant`] — and predicts **3.9%**, an order of
    /// magnitude under what the device does. **Shape**: narrowing `d_ff`
    /// (2048/4096/8192/10240) leaves the relative error flat at 36-39%, and
    /// narrowing the block to `768 x 3072`, the vision blocks' exact shape,
    /// still fails. **Weight resolution**: the two blocks quantize almost
    /// identically, 0.10-0.15 sigma per `uint8` step against 0.08.
    ///
    /// Why the vision blocks escape it: their `Erf` input spans
    /// `[-4.9, 5.3]`, and `erf` saturates past about +/-2, so their gate is
    /// effectively `0` or `2` and a badly computed `erf` still lands on the
    /// right value. The language model's input spans `[-1.4, 1.3]` — the
    /// transition region, where the result depends on the curve actually
    /// being the curve. A model is exposed to this bug exactly when its gate
    /// does not saturate.
    ///
    /// Two ways around it are already closed. Per-channel quantization: the
    /// provider refuses it outright — `Cannot support perchannel
    /// dequantize!`. Wider intermediates: 16-bit `QuantizeLinear` exists in
    /// opset 21, the runtime registers it, and the vendor libraries are full
    /// of `Activation16Bit` — but a graph carrying a single `uint16` *or*
    /// `int16` tensor is rejected whole with `ZHOUYI graph prepare error.
    /// Error code: 28`, while the identical graph at opset 21 with every
    /// tensor `uint8` compiles and runs. Those paths are internal to the
    /// toolchain and not reachable through ONNX.
    ///
    /// Until the activation is computed correctly, a language model's
    /// feed-forward block is not something to run through here at any speed.
    /// A vision block is.
    pub fn compile_gated_ffn(
        &self,
        ffn: &GatedFfn<'_>,
        max_tokens: usize,
        activation: GateActivation,
        calibration: &[f32],
    ) -> Result<CompiledLinear, OrtError> {
        let GatedFfn {
            d_model,
            d_ff,
            gate,
            up,
            down,
        } = *ffn;
        if d_model == 0 || d_ff == 0 || max_tokens == 0 {
            return Err(OrtError("a dimension is zero".into()));
        }
        for (name, weights, expected) in [
            ("gate", gate, d_ff * d_model),
            ("up", up, d_ff * d_model),
            ("down", down, d_model * d_ff),
        ] {
            if weights.len() != expected {
                return Err(OrtError(format!(
                    "{name} has {} weights, expected {expected}",
                    weights.len()
                )));
            }
        }
        if calibration.is_empty() || !calibration.len().is_multiple_of(d_model) {
            return Err(OrtError(format!(
                "calibration is {} values, not a whole number of {d_model}-feature rows",
                calibration.len()
            )));
        }

        // The block, computed on the host in `f32`, so every activation is
        // quantized to the range it actually occupies. A derived bound
        // cannot work here for the same reason it cannot work for a chain:
        // it compounds until the signal is gone.
        let rows = calibration.len() / d_model;
        let project = |x: &[f32], w: &[f32], in_dim: usize, out_dim: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; rows * out_dim];
            for r in 0..rows {
                for o in 0..out_dim {
                    let mut acc = 0.0f32;
                    for i in 0..in_dim {
                        acc += x[r * in_dim + i] * w[o * in_dim + i];
                    }
                    out[r * out_dim + o] = acc;
                }
            }
            out
        };

        let gate_out = project(calibration, gate, d_model, d_ff);
        let up_out = project(calibration, up, d_model, d_ff);

        // The intermediates the GELU spells out, named as the graph names
        // them so the two stay recognizably the same computation.
        let scaled: Vec<f32> = gate_out
            .iter()
            .map(|g| g * std::f32::consts::FRAC_1_SQRT_2)
            .collect();
        let erf_out: Vec<f32> = scaled.iter().map(|v| erf(*v)).collect();
        let shifted: Vec<f32> = erf_out.iter().map(|e| e + 1.0).collect();
        let gated: Vec<f32> = gate_out.iter().zip(&shifted).map(|(g, s)| g * s).collect();
        // `sigmoid(gate)`, for the `SigmoidOnly` diagnostic.
        let sig_plain: Vec<f32> = gate_out.iter().map(|g| 1.0 / (1.0 + (-g).exp())).collect();
        // `x * sigmoid(1.702x)`, spelled as the graph spells it.
        let sig_in: Vec<f32> = gate_out.iter().map(|g| g * GELU_SIGMOID_K).collect();
        let sig: Vec<f32> = sig_in.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        let sig_gated: Vec<f32> = gate_out.iter().zip(&sig).map(|(g, s)| g * s).collect();

        let activated: Vec<f32> = match activation {
            GateActivation::None => gate_out.clone(),
            // The graph stops at `gate * (1 + erf(..))`; the halving lives
            // in the `down` weights below.
            GateActivation::Gelu => gated.clone(),
            GateActivation::GeluSigmoid => sig_gated.clone(),
            GateActivation::ScaleOnly => sig_in.clone(),
            GateActivation::FanOut => gate_out.iter().zip(&sig_in).map(|(g, s)| g * s).collect(),
            GateActivation::SigmoidOnly => sig_plain.clone(),
            GateActivation::GeluSigmoidSplit
            | GateActivation::GeluSigmoidDup
            | GateActivation::GeluSigmoidTwoInput => sig_gated.clone(),
        };
        let hidden: Vec<f32> = activated.iter().zip(&up_out).map(|(a, u)| a * u).collect();

        let halved: Vec<f32>;
        let down = match activation {
            // `x * sigmoid(1.702x)` is already the whole activation, so
            // unlike the `Erf` spelling there is no stray factor to fold.
            GateActivation::None
            | GateActivation::GeluSigmoid
            | GateActivation::GeluSigmoidSplit
            | GateActivation::GeluSigmoidDup
            | GateActivation::GeluSigmoidTwoInput
            | GateActivation::ScaleOnly
            | GateActivation::SigmoidOnly
            | GateActivation::FanOut => down,
            GateActivation::Gelu => {
                halved = down.iter().map(|v| v * 0.5).collect();
                &halved
            }
        };
        let y = project(&hidden, down, d_ff, d_model);

        let weight_quant = |w: &[f32]| {
            let (lo, hi) = w
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            Quant::covering(lo, hi)
        };
        let quants = FfnQuants {
            x: observed_quant(calibration),
            gate: observed_quant(&gate_out),
            scaled: observed_quant(&scaled),
            erf: observed_quant(&erf_out),
            shifted: observed_quant(&shifted),
            gated: match activation {
                GateActivation::GeluSigmoid
                | GateActivation::GeluSigmoidSplit
                | GateActivation::GeluSigmoidDup
                | GateActivation::GeluSigmoidTwoInput => observed_quant(&sig_gated),
                GateActivation::ScaleOnly => observed_quant(&sig_in),
                GateActivation::SigmoidOnly => observed_quant(&sig_plain),
                GateActivation::FanOut => observed_quant(
                    &gate_out
                        .iter()
                        .zip(&sig_in)
                        .map(|(g, s)| g * s)
                        .collect::<Vec<f32>>(),
                ),
                _ => observed_quant(&gated),
            },
            sig_in: observed_quant(&sig_in),
            sig: match activation {
                GateActivation::SigmoidOnly => observed_quant(&sig_plain),
                _ => observed_quant(&sig),
            },
            up: observed_quant(&up_out),
            hidden: observed_quant(&hidden),
            y: observed_quant(&y),
            w_gate: weight_quant(gate),
            w_up: weight_quant(up),
            w_down: weight_quant(down),
        };

        let gate_q: Vec<u8> = gate.iter().map(|v| quants.w_gate.quantize(*v)).collect();
        let up_q: Vec<u8> = up.iter().map(|v| quants.w_up.quantize(*v)).collect();
        let down_q: Vec<u8> = down.iter().map(|v| quants.w_down.quantize(*v)).collect();

        let model = build_gated_ffn_model(
            &QuantizedFfn {
                d_model,
                d_ff,
                gate: &gate_q,
                up: &up_q,
                down: &down_q,
            },
            max_tokens,
            &quants,
            activation,
        );

        let context = ContextFile::new()?;
        let binary = self.compile(&model, &context)?;

        Ok(CompiledLinear {
            binary,
            k: d_model,
            n: d_model,
            max_tokens,
            inputs: match activation {
                GateActivation::GeluSigmoidTwoInput => 2,
                _ => 1,
            },
            x: quants.x,
            y: quants.y,
            unsmoothing: Vec::new(),
        })
    }

    /// Creates a session with EPContext enabled, then throws the session
    /// away and keeps what the provider wrote out.
    ///
    /// Session creation *is* compilation for this provider, so the session
    /// has already done its job by the time it exists.
    fn compile(&self, model: &[u8], context: &ContextFile) -> Result<Vec<u8>, OrtError> {
        let path = context
            .path
            .to_str()
            .ok_or_else(|| OrtError("context path is not UTF-8".into()))?;
        let settings = [
            ("ep.context_enable", "1"),
            // 1 embeds the binary in the file; 0 would write a second file
            // beside it and store only its name.
            ("ep.context_embed_mode", "1"),
            ("ep.context_file_path", path),
        ];

        let api = &self.inner.api;
        let mut options: *mut c_void = std::ptr::null_mut();
        let mut session: *mut c_void = std::ptr::null_mut();
        // SAFETY: every call takes a single out-pointer; the `CString`s and
        // `model` outlive the calls that read them.
        unsafe {
            let st = (api.create_session_options)(&mut options);
            if !st.is_null() {
                return Err(self.message(st, "CreateSessionOptions"));
            }
            for (key, value) in settings {
                let key_c = CString::new(key).map_err(|e| OrtError(e.to_string()))?;
                let value_c = CString::new(value).map_err(|e| OrtError(e.to_string()))?;
                let st = (api.add_session_config_entry)(options, key_c.as_ptr(), value_c.as_ptr());
                if !st.is_null() {
                    let e = self.message(st, key);
                    (api.release_session_options)(options);
                    return Err(e);
                }
            }
            let st = (self.append_zhouyi)(options);
            if !st.is_null() {
                let e = self.message(st, "AppendExecutionProvider_Zhouyi");
                (api.release_session_options)(options);
                return Err(e);
            }
            let st = (api.create_session_from_array)(
                self.inner.env,
                model.as_ptr().cast(),
                model.len(),
                options,
                &mut session,
            );
            (api.release_session_options)(options);
            if !st.is_null() {
                return Err(self.message(st, "CreateSessionFromArray"));
            }
            (api.release_session)(session);
        }

        let written = std::fs::read(&context.path).map_err(|e| {
            OrtError(format!(
                "the provider wrote no context model to {}: {e}",
                context.path.display()
            ))
        })?;
        extract_context_binary(&written)
    }
}

/// **Per-channel weight quantization is not available on this provider.**
///
/// One `uint8` scale per weight matrix is what limits a block's accuracy:
/// the 256 levels span the widest value anywhere in the tensor, so a fat
/// tail starves the bulk. Measured as the step in standard deviations of
/// `blk.0`'s gate, against the block's error — 0.097 -> 2.6% on gemma 4
/// E4B, 0.110 -> 6.5% on E2B, 0.155 -> 10.3% on qwen2 0.5B, 0.221 -> 12.1%
/// on llama 3.2 3B. Monotone across two architectures and four models, and
/// independent of the file's own quantization (the same llama measures
/// 12.5% at `Q4_K_M` and 12.1% at `Q8_0`).
///
/// One scale per output row is the textbook fix and the vendor's toolchain
/// advertises it (`just support perchannel quantization in dimension 0`).
/// It was implemented — 1-D scale and zero-point initializers on
/// `DequantizeLinear` with `axis = 0`, which is a `[n, k, 1, 1]` filter's
/// output-channel axis — and the provider **accepted the graph and computed
/// something else**: llama went 12.1% -> 79.4% and gemma 2.6% -> 104.7%,
/// while the artifact grew from 47 to 72 MiB and each run got a third
/// slower, which is what folding-failed-so-materialize looks like. The
/// provider's own libraries carry the string `Cannot support perchannel
/// dequantize!`, so the advertised support is for its own toolchain rather
/// than for a graph handed in through ONNX.
///
/// So per-tensor is not a choice here, and the accuracy ceiling it sets is
/// what `npu_tool`'s per-model gate exists to respect.
///
/// A quantization covering the values actually seen, with headroom.
///
/// The margin is for inputs the calibration did not reach, and it earns its
/// keep. Narrowing it looks like free precision — less range wasted, finer
/// steps — but measured on a real feed-forward block it went the other way:
/// 1.25 gave 10% worst-case error, 1.10 gave 18%, 1.02 gave 24%. Values
/// past the calibrated range saturate, and saturation costs far more than
/// coarseness. Calibrate on enough data that the range is real, then leave
/// room for the tail.
///
/// Clipping *inside* the observed extremes was tried too, since that
/// experiment only ever moved the range outward and so said nothing about
/// moving it inward. A per-side quantile in place of the observed maximum,
/// on `blk.0` of Gemma 4 E4B at 128 tokens, measured: 1.0 (the maximum,
/// i.e. what this does) 6.30 absolute error, 0.9999 5.90, 0.999 5.20, 0.99
/// 13.30 — against a signal magnitude of 16.06. So there is a shallow
/// optimum just inside the maximum worth about 17%, and it is nowhere near
/// enough to matter. No knob was kept for it.
/// The headroom [`observed_quant`] leaves past the values it saw, and the
/// share of the representable range [`row_gain`] therefore aims a row at.
///
/// One constant for both, because they are two halves of one decision: the
/// calibrated extreme sits at `1 / ACTIVATION_MARGIN` of what the `uint8`
/// grid can represent, so a row scaled to that same fraction lands exactly
/// where the widest calibration row landed — and its output lands inside
/// the output range calibrated from that row, with the same margin left
/// over. Aiming higher would buy a fraction of a bit on the input and risk
/// clipping the output, which costs far more.
const ACTIVATION_MARGIN: f32 = 1.25;

/// The most [`row_gain`] will scale a row by.
///
/// A row two orders of magnitude below the calibrated range is not a row
/// this can rescue: the scaled result would sit on a handful of levels of
/// whatever it feeds next, and a bound keeps a degenerate row — an
/// all-but-zero activation early in a prompt — from being amplified into
/// noise. 32 is five bits of recovery, which covers every gap measured on
/// this project's checkpoints (the worst was about 9x).
const MAX_ROW_GAIN: f32 = 32.0;

/// What to multiply one row of activations by so that it fills the range
/// this projection was calibrated for.
///
/// **This is the difference between a block that works and one that returns
/// nothing.** A projection on this device holds one `uint8` scale for the
/// whole activation tensor, and that scale comes from the widest rows in
/// the calibration capture. A row well below those extremes is then spread
/// over a handful of the 256 levels, and the damage compounds: the gate and
/// up projections lose most of their precision, their product loses it
/// again, and the `down` projection is handed something with almost no
/// signal left in it.
///
/// Measured on llama 3.2 3B `Q8_0`, device against backend on the model's
/// own activations, before this existed:
///
/// | `\|x\|` peak | rms error | cosine |
/// |---|---|---|
/// | 6.0 - 7.7 | 13 - 19% | 0.99 |
/// | 0.70 - 0.80 | 69 - 100% | 0.80 |
///
/// Both rows are the same blocks at the same width — the only variable is
/// how far the input fell below what the block was calibrated on. That is
/// why the failure looked like a decode failure for so long: a decode step
/// is one ordinary row, so it is *always* in the bottom row of that table,
/// while a 16-token prefill chunk usually contains something wide enough to
/// sit in the top one. It is not a property of decode at all.
///
/// The fix is exact rather than approximate, which is what makes it safe: a
/// projection is linear, so `W(gx) / g = Wx` for any `g`, and
/// [`NpuLinear::forward_into`] divides it straight back out. Each of the
/// three projections in a gated block picks its own `g` from its own input,
/// so nothing accumulates across them.
///
/// `lo` and `hi` are the row's extremes and `q_lo`/`q_hi` the interval the
/// quantization can represent. Both sides are checked because [`Quant`] is
/// asymmetric, and the tighter of the two wins. Never below `1.0`: a row
/// that already reaches the range is left exactly as it was, so this can
/// only ever be a no-op on the inputs that used to work.
/// Whether [`row_gain`] is applied at all — off with
/// `ORANGU_NPU_NO_ROW_GAIN=1`.
///
/// A control arm for the rescaling, so its contribution can be measured on
/// the same compiled artifacts rather than argued for.
fn row_gain_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("ORANGU_NPU_NO_ROW_GAIN").is_ok_and(|v| !matches!(v.as_str(), "" | "0"))
    })
}

/// Whether `ORANGU_NPU_GAIN_TRACE=1` asked for one line per row per
/// projection. Enormously noisy; a measurement tool only.
fn gain_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ORANGU_NPU_GAIN_TRACE").is_ok_and(|v| !matches!(v.as_str(), "" | "0"))
    })
}

fn row_gain(lo: f32, hi: f32, q_lo: f32, q_hi: f32) -> f32 {
    // A side with no values on it does not constrain the gain at all.
    let room = |v: f32, limit: f32| {
        if v == 0.0 {
            f32::INFINITY
        } else {
            (limit / v).abs()
        }
    };
    let gain = room(hi, q_hi).min(room(lo, q_lo)) / ACTIVATION_MARGIN;
    if gain.is_nan() {
        return 1.0;
    }
    // `clamp` carries an infinite gain (an all-zero row) down to the bound.
    gain.clamp(1.0, MAX_ROW_GAIN)
}

fn observed_quant(values: &[f32]) -> Quant {
    const MARGIN: f32 = ACTIVATION_MARGIN;
    let (lo, hi) = values
        .iter()
        .filter(|v| v.is_finite())
        .fold((0.0f32, 0.0f32), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    Quant::covering(lo * MARGIN, hi * MARGIN)
}

/// What sits on a gated feed-forward network's gate branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateActivation {
    /// No activation — the gate is multiplied in as it comes out of its
    /// projection. Not a real network, but the structure without the part
    /// most likely to be rejected, which makes it the right thing to try
    /// first when something does not compile.
    None,
    /// [`Self::GeluSigmoid`] with the gate projected twice from **two
    /// separate graph inputs** carrying the same activations.
    ///
    /// The one spelling of the duplicate the provider cannot undo.
    /// [`Self::GeluSigmoidDup`] duplicates the `Conv` node and comes back
    /// bit-identical to the broken version, because common-subexpression
    /// elimination merges the copies and rebuilds the diamond. Two declared
    /// inputs are distinct tensors, so there is nothing to merge.
    ///
    /// Costs the host one extra upload of the activation (`tokens x d_model`
    /// bytes, and the same buffer, so no extra allocation) and the device one
    /// extra gate projection.
    GeluSigmoidTwoInput,
    /// [`Self::GeluSigmoid`] with the gate **projected twice**, so no
    /// activation tensor feeds two consumers at all.
    ///
    /// The diamond is what fails: every piece works alone (see
    /// [`Self::SigmoidOnly`], [`Self::FanOut`], [`Self::ScaleOnly`]), and
    /// giving the two arms matching shapes does not help
    /// ([`Self::GeluSigmoidSplit`]). This removes the shared tensor instead
    /// of disguising it: two `Conv` nodes over the *same* weight
    /// initializer, one feeding the activation chain and one the multiply.
    /// The weights are shared, so it costs a fourth projection's arithmetic
    /// and no extra memory — affordable against a device otherwise 4.8x
    /// faster than the CPU on this block.
    GeluSigmoidDup,
    /// [`Self::GeluSigmoid`] with the diamond broken: the gate reaches the
    /// final multiply through its own `Mul` by one rather than directly.
    ///
    /// The bisection says the failure needs the gate read twice *and* a
    /// nonlinear op on one of the two branches — either alone is fine. This
    /// gives the other branch a `Mul` of its own, so the two arms of the
    /// diamond are the same shape, which is the cheapest thing that could
    /// separate them if the provider is fusing across the split.
    GeluSigmoidSplit,
    /// `sigmoid(gate)`, with the gate read **once**. **A diagnostic, not an
    /// activation.**
    ///
    /// Splits the two things [`Self::GeluSigmoid`] does at the same time.
    /// That spelling reads the gate twice *and* puts a nonlinear op in the
    /// chain; [`Self::FanOut`] showed the double read alone is fine, so this
    /// asks the other half — whether the provider's `Sigmoid` is itself
    /// wrong — by using it with no second read at all.
    SigmoidOnly,
    /// `gate * (gate * 1.702)`. **A diagnostic, not an activation.**
    ///
    /// [`Self::ScaleOnly`] with the gate read a second time, and nothing
    /// else added. Both real activations read the dequantized gate twice —
    /// once into the activation's own chain, once into the multiply that
    /// applies it — and [`Self::None`] and [`Self::ScaleOnly`] read it once.
    /// That is the only structural difference left between the spellings
    /// that fail and the ones that pass.
    FanOut,
    /// `gate * 1.702`, and nothing else. **A diagnostic, not an
    /// activation.**
    ///
    /// Isolates one mechanism: the scalar constant, which this graph encodes
    /// as a `uint8` `1` whose *scale* is the value. Both real activations
    /// use it and [`Self::None`] does not, which is exactly the line the
    /// measurements fall along.
    ScaleOnly,
    /// GELU as `x * sigmoid(1.702 * x)`.
    ///
    /// The *approximation* — max absolute error about 0.02 against true
    /// GELU — and on this device by far the more accurate of the two, which
    /// is not the ordering the arithmetic suggests. [`Self::Gelu`] spells
    /// the exact formula and then hands its shape to the provider's `Erf`,
    /// which turns out to be the inaccurate part: a block whose `Erf` input
    /// stays inside the transition region (roughly +/-2, where `erf`
    /// actually varies) comes back about 36% wrong, while one whose input
    /// saturates is fine because a saturated `erf` is +/-1 no matter how
    /// badly it is computed. That is why Gemma 4's vision blocks pass with
    /// [`Self::Gelu`] and its language-model blocks do not — see
    /// [`NpuOrt::compile_gated_ffn`].
    ///
    /// `Sigmoid` is registered in the provider's own builder table, same as
    /// `Erf`. The difference is only in how well each is implemented.
    GeluSigmoid,
    /// GELU, exactly: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    ///
    /// Spelled out because the provider registers no `Gelu` builder — the
    /// string exists in the library, but the registration table maps only
    /// `Erf`, `Add` and `Mul`, so those are what it gets. Two of the five
    /// operations that formula implies are folded into dequantization
    /// scales and cost nothing.
    Gelu,
}

/// The weights of a gated feed-forward block, `down(act(gate(x)) * up(x))`.
///
/// `gate` and `up` are row-major `[d_ff][d_model]`; `down` is
/// `[d_model][d_ff]` — the layout GGUF already stores them in.
pub struct GatedFfn<'a> {
    pub d_model: usize,
    pub d_ff: usize,
    pub gate: &'a [f32],
    pub up: &'a [f32],
    pub down: &'a [f32],
}

/// The same weights, quantized, as the graph builder wants them.
struct QuantizedFfn<'a> {
    d_model: usize,
    d_ff: usize,
    gate: &'a [u8],
    up: &'a [u8],
    down: &'a [u8],
}

/// Every activation range a gated FFN needs quantized, measured on the host.
struct FfnQuants {
    x: Quant,
    gate: Quant,
    scaled: Quant,
    erf: Quant,
    shifted: Quant,
    /// `1.702 * gate`, and the sigmoid of it — the `GeluSigmoid` spelling's
    /// two intermediates, unused by the others.
    sig_in: Quant,
    sig: Quant,
    gated: Quant,
    up: Quant,
    hidden: Quant,
    y: Quant,
    w_gate: Quant,
    w_up: Quant,
    w_down: Quant,
}

/// The logistic approximation's coefficient: `gelu(x) ~= x * sigmoid(kx)`.
const GELU_SIGMOID_K: f32 = 1.702;

/// Emits `down(act(gate(x)) * up(x))` as one graph.
///
/// This is the shape of the feed-forward block in both Gemma 4 models, and
/// the reason it is worth expressing at all: the FFN is the large majority
/// of a transformer layer's arithmetic, and as three separate graphs it pays
/// the host boundary three times.
///
/// Note the diamond — `gate` and `up` both read the same input, and `Mul`
/// joins them. That is not something a chain of projections can express, and
/// it is why this has its own builder rather than another case of
/// [`build_stack_model`].
fn build_gated_ffn_model(
    w: &QuantizedFfn,
    m: usize,
    quants: &FfnQuants,
    activation: GateActivation,
) -> Vec<u8> {
    let (d_model, d_ff) = (w.d_model, w.d_ff);
    let attrs = projection_attrs();
    let mut g = QdqGraph::new();

    let x = g.dequantize("X_q", quants.x);
    let gate_w = g.constant(w.gate, &[d_ff as i64, d_model as i64, 1, 1], quants.w_gate);
    let up_w = g.constant(w.up, &[d_ff as i64, d_model as i64, 1, 1], quants.w_up);
    let down_w = g.constant(w.down, &[d_model as i64, d_ff as i64, 1, 1], quants.w_down);

    let gate_u = g.op_quantized("Conv", &[&x, &gate_w], &attrs, quants.gate);
    let activated = match activation {
        GateActivation::None => g.dequantize(&gate_u, quants.gate),
        // `gate * sigmoid(1.702 * gate)`. Three operations against the `Erf`
        // spelling's four, and — the reason it exists — no `Erf`.
        GateActivation::SigmoidOnly => {
            let gate = g.dequantize(&gate_u, quants.gate);
            g.op("Sigmoid", &[&gate], &[], quants.sig)
        }
        // Two `Conv` nodes over the same weight initializer. `gate_u` is
        // deliberately not reused here: reusing it *is* the diamond.
        // The gate's second projection comes from `X2_q`, a *separate*
        // graph input the host binds to the same activations. Nothing here
        // is a duplicate the provider can merge away.
        GateActivation::GeluSigmoidTwoInput => {
            let gate_a = g.dequantize(&gate_u, quants.gate);
            let x2 = g.dequantize("X2_q", quants.x);
            let gate_b_u = g.op_quantized("Conv", &[&x2, &gate_w], &attrs, quants.gate);
            let gate_b = g.dequantize(&gate_b_u, quants.gate);
            let k = g.constant(
                &[1u8],
                &[1],
                Quant {
                    scale: GELU_SIGMOID_K,
                    zero_point: 0,
                },
            );
            let scaled = g.op("Mul", &[&gate_b, &k], &[], quants.sig_in);
            let sig = g.op("Sigmoid", &[&scaled], &[], quants.sig);
            g.op("Mul", &[&gate_a, &sig], &[], quants.gated)
        }
        GateActivation::GeluSigmoidDup => {
            let gate_a = g.dequantize(&gate_u, quants.gate);
            let gate_b_u = g.op_quantized("Conv", &[&x, &gate_w], &attrs, quants.gate);
            let gate_b = g.dequantize(&gate_b_u, quants.gate);
            let k = g.constant(
                &[1u8],
                &[1],
                Quant {
                    scale: GELU_SIGMOID_K,
                    zero_point: 0,
                },
            );
            let scaled = g.op("Mul", &[&gate_b, &k], &[], quants.sig_in);
            let sig = g.op("Sigmoid", &[&scaled], &[], quants.sig);
            g.op("Mul", &[&gate_a, &sig], &[], quants.gated)
        }
        GateActivation::GeluSigmoidSplit => {
            let gate = g.dequantize(&gate_u, quants.gate);
            let k = g.constant(
                &[1u8],
                &[1],
                Quant {
                    scale: GELU_SIGMOID_K,
                    zero_point: 0,
                },
            );
            let one = g.constant(
                &[1u8],
                &[1],
                Quant {
                    scale: 1.0,
                    zero_point: 0,
                },
            );
            let scaled = g.op("Mul", &[&gate, &k], &[], quants.sig_in);
            let sig = g.op("Sigmoid", &[&scaled], &[], quants.sig);
            // The gate's own arm, through a `Mul` so both arms of the
            // diamond leave the gate the same way.
            let passthrough = g.op("Mul", &[&gate, &one], &[], quants.gate);
            g.op("Mul", &[&passthrough, &sig], &[], quants.gated)
        }
        GateActivation::FanOut => {
            let gate = g.dequantize(&gate_u, quants.gate);
            let k = g.constant(
                &[1u8],
                &[1],
                Quant {
                    scale: GELU_SIGMOID_K,
                    zero_point: 0,
                },
            );
            let scaled = g.op("Mul", &[&gate, &k], &[], quants.sig_in);
            g.op("Mul", &[&gate, &scaled], &[], quants.gated)
        }
        GateActivation::ScaleOnly => {
            let gate = g.dequantize(&gate_u, quants.gate);
            let k = g.constant(
                &[1u8],
                &[1],
                Quant {
                    scale: GELU_SIGMOID_K,
                    zero_point: 0,
                },
            );
            g.op("Mul", &[&gate, &k], &[], quants.gated)
        }
        GateActivation::GeluSigmoid => {
            let gate = g.dequantize(&gate_u, quants.gate);
            let k = g.constant(
                &[1u8],
                &[1],
                Quant {
                    scale: GELU_SIGMOID_K,
                    zero_point: 0,
                },
            );
            let scaled = g.op("Mul", &[&gate, &k], &[], quants.sig_in);
            let sig = g.op("Sigmoid", &[&scaled], &[], quants.sig);
            g.op("Mul", &[&gate, &sig], &[], quants.gated)
        }
        GateActivation::Gelu => {
            let gate = g.dequantize(&gate_u, quants.gate);
            // Constants come in as a `uint8` 1 whose scale *is* the value,
            // so `(1 - 0) * scale` dequantizes to exactly what is wanted and
            // no float initializer is needed.
            //
            // These two multiplies look like they could be folded into a
            // dequantization scale instead — read the same tensor twice, once
            // scaled — and that is free where it works. It does not work
            // here: a QDQ tensor carries one quantization, and giving it two
            // produced a block that compiled and then computed something
            // else, 91% wrong against a CPU reference. Spending two
            // operations to stay inside the format is the trade.
            let inv_sqrt2 = g.constant(
                &[1u8],
                &[1],
                Quant {
                    scale: std::f32::consts::FRAC_1_SQRT_2,
                    zero_point: 0,
                },
            );
            let scaled = g.op("Mul", &[&gate, &inv_sqrt2], &[], quants.scaled);
            let erf = g.op("Erf", &[&scaled], &[], quants.erf);
            let one = g.constant(
                &[1u8],
                &[1],
                Quant {
                    scale: 1.0,
                    zero_point: 0,
                },
            );
            let shifted = g.op("Add", &[&erf, &one], &[], quants.shifted);
            // GELU's remaining `* 0.5` is not here: it is folded into the
            // `down` weights, which are quantized anyway, so it is exact and
            // costs neither an operation nor a quantization stage.
            g.op("Mul", &[&gate, &shifted], &[], quants.gated)
        }
    };

    let up = g.op("Conv", &[&x, &up_w], &attrs, quants.up);
    let hidden = g.op("Mul", &[&activated, &up], &[], quants.hidden);
    g.op_into("Conv", &[&hidden, &down_w], &attrs, "Y_q", quants.y);

    let shape = [1, d_model as i64, m as i64, 1];
    let out = ("Y_q", &shape[..]);
    match activation {
        GateActivation::GeluSigmoidTwoInput => {
            g.finish_inputs(&[("X_q", &shape), ("X2_q", &shape)], out)
        }
        _ => g.finish(("X_q", &shape), out),
    }
}

/// `out[i] = silu(gate[i]) * up[i]` — the SwiGLU half, for llama, mistral,
/// qwen and phi.
///
/// `silu(x)` is `x * sigmoid(x)`, which is one exponential, so the vector
/// path is the same shape as [`gelu_mul_into`]'s and shares its `exp`.
fn silu_mul_into(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64, and the three lengths are equal
    // and are the bound the helper stays within.
    unsafe {
        silu_mul_neon(gate, up, out)
    };
    #[cfg(not(target_arch = "aarch64"))]
    silu_mul_scalar(gate, up, out);
}

/// [`silu_mul_into`]'s portable form, and the reference its vector path is
/// checked against.
fn silu_mul_scalar(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((o, g), u) in out.iter_mut().zip(gate).zip(up) {
        *o = (*g / (1.0 + (-*g).exp())) * u;
    }
}

/// [`silu_mul_into`] on NEON.
#[cfg(target_arch = "aarch64")]
unsafe fn silu_mul_neon(gate: &[f32], up: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;
    unsafe {
        let n = out.len();
        let mut i = 0;
        while i + 4 <= n {
            let g = vld1q_f32(gate.as_ptr().add(i));
            let u = vld1q_f32(up.as_ptr().add(i));
            // `g / (1 + e^-g)`
            let e = exp_neon(vnegq_f32(g));
            let sig = vdivq_f32(g, vaddq_f32(vdupq_n_f32(1.0), e));
            vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(sig, u));
            i += 4;
        }
        if i < n {
            silu_mul_scalar(&gate[i..], &up[i..], &mut out[i..]);
        }
    }
}

/// `out[i] = gelu(gate[i]) * up[i]`, the host half of a split feed-forward
/// block.
///
/// Measured at **20% of prefill CPU** in the scalar form — 10.9% in the loop
/// and 8.9% in `expf` beneath it — which makes it the largest single cost
/// this side of the boundary once the projections moved to the device. It is
/// `d_ff` values per token per layer: 10240 x 16 x 42 for one chunk of
/// Gemma 4 E4B, so a scalar `expf` call each is 6.9 million of them.
fn gelu_mul_into(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64, and the three lengths are equal
    // and are the bound the helper stays within.
    unsafe {
        gelu_mul_neon(gate, up, out)
    };
    #[cfg(not(target_arch = "aarch64"))]
    gelu_mul_scalar(gate, up, out);
}

/// [`gelu_mul_into`]'s portable form, and the reference its vector path is
/// checked against.
fn gelu_mul_scalar(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((o, g), u) in out.iter_mut().zip(gate).zip(up) {
        *o = gelu(*g) * u;
    }
}

/// `e^x` for four lanes.
///
/// Range reduction to `x = n*ln2 + r` with `|r| <= ln2/2`, a degree-5
/// polynomial for `e^r`, and `2^n` folded in by adding `n` to the exponent
/// field. Accurate to about one ulp, which is far tighter than anything
/// downstream needs: the result is multiplied by an activation that has
/// already been through a `uint8` quantization.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn exp_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        // Clamped so the exponent arithmetic cannot overflow into a NaN for
        // inputs far outside the range `erf` ever produces.
        let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-88.0)), vdupq_n_f32(88.0));
        let n = vrndnq_f32(vmulq_n_f32(x, std::f32::consts::LOG2_E));
        // Two-part `ln2` so the reduction keeps its low bits.
        let r = vfmaq_n_f32(x, n, -0.693_359_4);
        let r = vfmaq_n_f32(r, n, 2.121_944_4e-4);
        let mut p = vfmaq_n_f32(vdupq_n_f32(1.0 / 24.0), r, 1.0 / 120.0);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), r, p);
        p = vfmaq_f32(vdupq_n_f32(0.5), r, p);
        p = vfmaq_f32(vdupq_n_f32(1.0), r, p);
        p = vfmaq_f32(vdupq_n_f32(1.0), r, p);
        // `2^n`, built by placing `n` in the exponent field directly.
        let scale = vreinterpretq_f32_s32(vshlq_n_s32(
            vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)),
            23,
        ));
        vmulq_f32(p, scale)
    }
}

/// [`gelu_mul_into`] on NEON.
///
/// `erf` is Abramowitz and Stegun 7.1.26, the same formula [`erf`] uses, so
/// the vector and scalar paths agree to their shared approximation rather
/// than to two different ones.
#[cfg(target_arch = "aarch64")]
unsafe fn gelu_mul_neon(gate: &[f32], up: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;
    unsafe {
        let n = out.len();
        let mut i = 0;
        while i + 4 <= n {
            let g = vld1q_f32(gate.as_ptr().add(i));
            let u = vld1q_f32(up.as_ptr().add(i));

            let x = vmulq_n_f32(g, std::f32::consts::FRAC_1_SQRT_2);
            let ax = vabsq_f32(x);
            let t = vdivq_f32(
                vdupq_n_f32(1.0),
                vfmaq_n_f32(vdupq_n_f32(1.0), ax, 0.327_591_1),
            );
            let mut poly = vfmaq_n_f32(vdupq_n_f32(-1.453_152), t, 1.061_405_4);
            poly = vfmaq_f32(vdupq_n_f32(1.421_413_7), t, poly);
            poly = vfmaq_f32(vdupq_n_f32(-0.284_496_74), t, poly);
            poly = vfmaq_f32(vdupq_n_f32(0.254_829_6), t, poly);
            poly = vmulq_f32(poly, t);
            let e = exp_neon(vnegq_f32(vmulq_f32(ax, ax)));
            let mag = vsubq_f32(vdupq_n_f32(1.0), vmulq_f32(poly, e));
            // `erf` is odd: the sign of `x` carries through.
            let erf = vbslq_f32(vcltq_f32(x, vdupq_n_f32(0.0)), vnegq_f32(mag), mag);

            let gelu = vmulq_f32(vmulq_n_f32(g, 0.5), vaddq_f32(vdupq_n_f32(1.0), erf));
            vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(gelu, u));
            i += 4;
        }
        if i < n {
            gelu_mul_scalar(&gate[i..], &up[i..], &mut out[i..]);
        }
    }
}

/// GELU, as [`GateActivation::Gelu`] defines it.
///
/// The reference the compiled block is checked against, and — since the
/// activation moved off the device — the function
/// [`NpuGatedFfn::forward_into`] actually applies between the projections.
/// [`NpuOrt::compile_gated_ffn`] still does not call it: that one needs each
/// intermediate separately, to quantize every one to its own range.
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// `erf`, to about seven digits — Abramowitz and Stegun 7.1.26.
///
/// Written out because Rust's standard library has no `erf` and this needs
/// only to be good enough to *calibrate* against: the device computes its
/// own, and the reference this feeds is compared with a tolerance set by the
/// quantization step, which is orders of magnitude coarser.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

/// `count` output-channel indices spread evenly over `n`, ends included.
///
/// Evenly rather than the first `count`: a matrix's channels are not
/// interchangeable, and scoring only the ones at the front would pick
/// `alpha` for a corner of it.
fn sampled_channels(n: usize, count: usize) -> Vec<usize> {
    if n <= count {
        return (0..n).collect();
    }
    (0..count)
        .map(|i| i * (n - 1) / (count - 1).max(1))
        .collect()
}

/// The largest output magnitude a projection can produce for any input
/// inside `range`.
///
/// Derived rather than measured: every product is bounded by the extremes of
/// the two operands, so accumulating the larger corner product per input
/// feature gives a bound the layer cannot exceed. Deliberately pessimistic —
/// real activations cancel — but a bound that cannot saturate, since
/// saturation destroys a result silently where coarseness only blunts it.
fn worst_case_magnitude(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    range: (f32, f32),
) -> f32 {
    let mut bound = 0.0f32;
    for col in 0..out_features {
        let mut acc = 0.0f32;
        for row in 0..in_features {
            let w = weights[col * in_features + row];
            acc += (range.0 * w).abs().max((range.1 * w).abs());
        }
        bound = bound.max(acc);
    }
    bound
}

/// A scratch file for the provider to write its context model into, removed
/// when this goes out of scope.
///
/// ONNX Runtime's EPContext mechanism only writes to a path — there is no
/// in-memory variant — so compiling means touching the filesystem whether
/// the caller wanted to or not. The name carries the process id and a
/// counter so that concurrent compiles cannot collide.
struct ContextFile {
    path: std::path::PathBuf,
}

impl ContextFile {
    fn new() -> Result<Self, OrtError> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let name = format!(
            "orangu-npu-{}-{}.onnx",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(name);
        // Removed up front rather than trusted to be absent: the provider
        // refuses to regenerate over an existing context file.
        let _ = std::fs::remove_file(&path);
        Ok(Self { path })
    }
}

impl Drop for ContextFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Pulls the compiled binary out of an EPContext model.
///
/// The file is an ONNX protobuf whose single node carries the compiled
/// graph in its `ep_cache_context` string attribute. Rather than parse the
/// whole message, this finds the attribute by name and reads the
/// length-delimited field that follows it — the attribute value is
/// `AttributeProto.s`, field 4, wire type 2, so the tag byte is `0x22`.
fn extract_context_binary(model: &[u8]) -> Result<Vec<u8>, OrtError> {
    const NAME: &[u8] = b"ep_cache_context";
    let at = model
        .windows(NAME.len())
        .position(|w| w == NAME)
        .ok_or_else(|| OrtError("context model has no ep_cache_context attribute".into()))?
        + NAME.len();

    if model.get(at) != Some(&0x22) {
        return Err(OrtError(
            "ep_cache_context is not followed by a length-delimited value".into(),
        ));
    }

    let (mut cursor, mut len, mut shift) = (at + 1, 0usize, 0u32);
    loop {
        let byte = *model
            .get(cursor)
            .ok_or_else(|| OrtError("truncated ep_cache_context length".into()))?;
        cursor += 1;
        if shift > 56 {
            return Err(OrtError("ep_cache_context length is not a varint".into()));
        }
        len |= ((byte & 0x7f) as usize) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            break;
        }
    }

    let end = cursor
        .checked_add(len)
        .filter(|e| *e <= model.len())
        .ok_or_else(|| OrtError("ep_cache_context runs past the end of the model".into()))?;
    let binary = &model[cursor..end];

    // The provider is documented to embed a path here when
    // `ep.context_embed_mode` is 0. Compilation asks for 1, so anything
    // other than an executable means the setting did not take.
    if !binary.starts_with(b"\x7fELF") {
        return Err(OrtError(
            "ep_cache_context does not hold an executable; embed mode may not have applied".into(),
        ));
    }
    Ok(binary.to_vec())
}

/// Which activation the host applies between the projections.
///
/// The activation is the one part of a gated block this device gets wrong
/// (see [`NpuOrt::compile_gated_ffn`]), so it runs on the host — which makes
/// supporting a second one a matter of choosing a function rather than
/// finding ops the provider computes correctly. Gemma's blocks are GELU;
/// llama, mistral, qwen and phi are SiLU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostActivation {
    Gelu,
    Silu,
}

impl HostActivation {
    /// The activation itself, for a caller building its own reference.
    pub fn apply_host(self, x: f32) -> f32 {
        self.apply(x)
    }

    fn apply(self, x: f32) -> f32 {
        match self {
            HostActivation::Gelu => gelu(x),
            HostActivation::Silu => x / (1.0 + (-x).exp()),
        }
    }

    fn tag(self) -> u8 {
        match self {
            HostActivation::Gelu => 0,
            HostActivation::Silu => 1,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(HostActivation::Gelu),
            1 => Some(HostActivation::Silu),
            _ => None,
        }
    }
}

/// A gated feed-forward block compiled as three projections.
///
/// See [`NpuOrt::compile_gated_ffn_host`] for why it is three graphs and not
/// one. Serializes as its three parts back to back, so the format follows
/// [`CompiledLinear`]'s and gains nothing of its own to get wrong.
pub struct CompiledGatedFfn {
    pub gate: CompiledLinear,
    pub up: CompiledLinear,
    pub down: CompiledLinear,
    /// What the host applies between `gate`/`up` and `down`. Carried in the
    /// artifact because the serving process binds a block without ever
    /// seeing the model it came from.
    pub activation: HostActivation,
}

/// Several compiled projections that share one input, with nothing between
/// them — what `Q`, `K` and `V` are.
///
/// A list rather than three named fields, because the shape is "some
/// projections over one activation" and a family with a different count
/// (fused `QKV`, or a model without a separate `V`) should not need a new
/// type. Serializes as its parts back to back, following
/// [`CompiledLinear`]'s format exactly as [`CompiledGatedFfn`] does.
pub struct CompiledProjections {
    pub parts: Vec<CompiledLinear>,
}

/// Identifies a serialized [`CompiledProjections`].
pub(crate) const PROJ_MAGIC: &[u8] = b"ORANGU-NPU-PROJ-1";

impl CompiledProjections {
    /// Total bytes of compiled executable across the parts.
    pub fn binary_len(&self) -> usize {
        self.parts.iter().map(|p| p.binary().len()).sum()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = PROJ_MAGIC.to_vec();
        out.extend_from_slice(&(self.parts.len() as u32).to_le_bytes());
        for part in &self.parts {
            let bytes = part.to_bytes();
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&bytes);
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OrtError> {
        if !bytes.starts_with(PROJ_MAGIC) {
            return Err(OrtError("not a compiled NPU projection set".into()));
        }
        let mut at = PROJ_MAGIC.len();
        let count = bytes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize)
            .ok_or_else(|| OrtError("projection artifact is truncated".into()))?;
        at += 4;
        let mut parts = Vec::with_capacity(count);
        for _ in 0..count {
            let len = bytes
                .get(at..at + 4)
                .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize)
                .ok_or_else(|| OrtError("projection artifact is truncated".into()))?;
            at += 4;
            let part = bytes
                .get(at..at + len)
                .ok_or_else(|| OrtError("projection artifact is truncated".into()))?;
            parts.push(CompiledLinear::from_bytes(part)?);
            at += len;
        }
        Ok(Self { parts })
    }

    /// Loads every graph onto the device.
    pub fn bind(&self, runtime: &crate::npu::NpuRuntime) -> Result<Vec<NpuLinear>, OrtError> {
        self.parts.iter().map(|p| p.bind(runtime)).collect()
    }
}

/// Identifies a serialized [`CompiledGatedFfn`].
///
/// **Bumped to `-9` when the calibration row count stopped being the graph
/// width** (`npu_tool::CALIBRATION_ROWS`). Nothing about the *format*
/// changed; what changed is what the bytes mean, and an artifact compiled
/// from a one-row sample is not one this build would produce. The cache
/// keys on the calibration capture's digest, not on how many of its rows a
/// compile read, so the magic is the only thing that can tell the two
/// apart. Blocks compiled by an older build are read as absent and
/// recompiled.
pub(crate) const FFN_MAGIC: &[u8] = b"ORANGU-NPU-FFN-10";

impl CompiledGatedFfn {
    /// `d_model` in, `d_model` out.
    pub fn in_features(&self) -> usize {
        self.gate.in_features()
    }

    pub fn max_tokens(&self) -> usize {
        self.gate.max_tokens()
    }

    /// Total bytes of compiled executable across the three graphs.
    pub fn binary_len(&self) -> usize {
        self.gate.binary().len() + self.up.binary().len() + self.down.binary().len()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = FFN_MAGIC.to_vec();
        out.push(self.activation.tag());
        for part in [&self.gate, &self.up, &self.down] {
            let bytes = part.to_bytes();
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&bytes);
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OrtError> {
        if !bytes.starts_with(FFN_MAGIC) {
            return Err(OrtError("not a compiled NPU feed-forward block".into()));
        }
        let mut at = FFN_MAGIC.len();
        let activation = bytes
            .get(at)
            .copied()
            .and_then(HostActivation::from_tag)
            .ok_or_else(|| OrtError("artifact names an activation this cannot apply".into()))?;
        at += 1;
        let mut parts = Vec::with_capacity(3);
        for _ in 0..3 {
            let len_bytes = bytes
                .get(at..at + 4)
                .ok_or_else(|| OrtError("feed-forward artifact is truncated".into()))?;
            let len = u32::from_le_bytes(len_bytes.try_into().expect("4 bytes")) as usize;
            at += 4;
            let part = bytes
                .get(at..at + len)
                .ok_or_else(|| OrtError("feed-forward artifact is truncated".into()))?;
            parts.push(CompiledLinear::from_bytes(part)?);
            at += len;
        }
        let mut parts = parts.into_iter();
        Ok(Self {
            activation,
            gate: parts.next().expect("three parts"),
            up: parts.next().expect("three parts"),
            down: parts.next().expect("three parts"),
        })
    }

    /// Loads all three graphs onto the device.
    pub fn bind(&self, runtime: &crate::npu::NpuRuntime) -> Result<NpuGatedFfn, OrtError> {
        Ok(NpuGatedFfn {
            activation: self.activation,
            gate: self.gate.bind(runtime)?,
            up: self.up.bind(runtime)?,
            down: self.down.bind(runtime)?,
            scratch: std::cell::RefCell::new(FfnScratch::default()),
        })
    }
}

/// [`CompiledGatedFfn`] on the device.
pub struct NpuGatedFfn {
    activation: HostActivation,
    gate: NpuLinear,
    up: NpuLinear,
    down: NpuLinear,
    scratch: std::cell::RefCell<FfnScratch>,
}

/// [`NpuGatedFfn`]'s reusable working memory, so a serving loop allocates
/// nothing after the first call.
#[derive(Default)]
struct FfnScratch {
    gate: Vec<f32>,
    up: Vec<f32>,
    hidden: Vec<f32>,
}

impl NpuGatedFfn {
    pub fn in_features(&self) -> usize {
        self.gate.in_features()
    }

    pub fn max_tokens(&self) -> usize {
        self.gate.max_tokens()
    }

    /// Runs `down(gelu(gate(x)) * up(x))`.
    ///
    /// Three device inferences with one host step between the second and
    /// third. The activation is `f32` here and exact, which is the whole
    /// point: it is the one part of the block this device gets wrong.
    ///
    /// **`gate` and `up` are dispatched before either is waited for.** They
    /// read the same activations and neither reads the other, so there is
    /// nothing to order them by except the driver's own queue — and left to
    /// itself the driver runs them back to back. `UMD_LOG_LEVEL=3` on a
    /// width-1 block shows exactly that, three jobs and three waits:
    ///
    /// ```text
    /// 18:31:11:188513 dispatch job: 0x100000001
    /// 18:31:11:189554 job: 0x100000001 status is DONE   (1041 us)
    /// 18:31:11:189646 dispatch job: 0x200000001
    /// 18:31:11:190644 job: 0x200000001 status is DONE   ( 998 us)
    /// 18:31:11:190747 dispatch job: 0x300000001
    /// 18:31:11:191651 job: 0x300000001 status is DONE   ( 904 us)
    /// ```
    ///
    /// A job of that shape moves about 25 MB of `uint8` weights, so one of
    /// them is running at roughly 24 GB/s — under what this memory system
    /// can do, which is what says there is room for two at once rather than
    /// just a longer queue. `down` still has to wait: it reads the
    /// activation of the other two.
    pub fn forward_into(
        &self,
        x: &[f32],
        tokens: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), OrtError> {
        let scratch = &mut *self.scratch.borrow_mut();
        self.gate.dispatch(x, tokens)?;
        self.up.dispatch(x, tokens)?;
        self.gate.collect(tokens, &mut scratch.gate)?;
        self.up.collect(tokens, &mut scratch.up)?;

        // Resized rather than cleared and re-extended: this buffer is the
        // same length on every call, so `resize` is a no-op after the first
        // and the vector path writes every element anyway.
        // Resized rather than cleared and re-extended: this buffer is the
        // same length on every call, so `resize` is a no-op after the first
        // and the activation writes every element anyway.
        scratch.hidden.resize(scratch.gate.len(), 0.0);
        match self.activation {
            HostActivation::Gelu => gelu_mul_into(&scratch.gate, &scratch.up, &mut scratch.hidden),
            HostActivation::Silu => silu_mul_into(&scratch.gate, &scratch.up, &mut scratch.hidden),
        }
        self.down.forward_into(&scratch.hidden, tokens, out)
    }

    /// [`Self::forward_into`], allocating the result.
    pub fn forward(&self, x: &[f32], tokens: usize) -> Result<Vec<f32>, OrtError> {
        let mut out = Vec::new();
        self.forward_into(x, tokens, &mut out)?;
        Ok(out)
    }
}

/// Identifies a serialized [`CompiledLinear`]. The trailing digit is a
/// format version: a reader that does not recognize it should refuse rather
/// than guess at a layout.
pub(crate) const ARTIFACT_MAGIC: &[u8] = b"ORANGU-NPU-LINEAR-3";

/// Magic, three `u32` dimensions, the input count, two `(f32, u32)`
/// quantizations, and the executable's length.
const ARTIFACT_HEADER: usize = ARTIFACT_MAGIC.len() + 4 * 4 + 8 * 2 + 4 + 4;

/// One compiled linear projection: the AIPU executable, plus the
/// quantization the graph was built around.
///
/// Compiling is the expensive step and this is its whole result, so it is
/// deliberately plain data. Bind it to a runtime to run it; keep `binary`
/// if you would rather compile now and run later, or elsewhere.
pub struct CompiledLinear {
    binary: Vec<u8>,
    k: usize,
    n: usize,
    max_tokens: usize,
    /// How many input tensors the graph declares.
    ///
    /// Two for [`GateActivation::GeluSigmoidTwoInput`], which binds the same
    /// activations to both; one for everything else. Carried in the artifact
    /// because the serving process binds the graph without ever seeing how
    /// it was built.
    inputs: usize,
    x: Quant,
    y: Quant,
    /// The reciprocal of the per-input-channel scale folded into the
    /// weights, or empty when none was. Applied to `x` on the way in — see
    /// [`NpuOrt::smoothing_scales`].
    unsmoothing: Vec<f32>,
}

impl CompiledLinear {
    /// The compiled AIPU executable.
    pub fn binary(&self) -> &[u8] {
        &self.binary
    }

    pub fn in_features(&self) -> usize {
        self.k
    }

    pub fn out_features(&self) -> usize {
        self.n
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Serializes this into a self-describing artifact.
    ///
    /// The executable alone is not enough to use the layer: the scales it
    /// was compiled around live here and nowhere else, so quantizing an
    /// input against a guess would produce plausible, wrong numbers. Keeping
    /// them in one artifact makes that mistake impossible to make quietly.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(ARTIFACT_HEADER + self.binary.len());
        out.extend_from_slice(ARTIFACT_MAGIC);
        for value in [
            self.k as u32,
            self.n as u32,
            self.max_tokens as u32,
            self.inputs as u32,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for quant in [self.x, self.y] {
            out.extend_from_slice(&quant.scale.to_le_bytes());
            out.extend_from_slice(&(quant.zero_point as u32).to_le_bytes());
        }
        out.extend_from_slice(&(self.unsmoothing.len() as u32).to_le_bytes());
        for value in &self.unsmoothing {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&(self.binary.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.binary);
        out
    }

    /// Reads back what [`Self::to_bytes`] wrote.
    ///
    /// This is the entry point for a serving process, which must never have
    /// opened [`NpuOrt`] — see the module doc on why compiling and serving
    /// are separate processes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OrtError> {
        if bytes.len() < ARTIFACT_HEADER || !bytes.starts_with(ARTIFACT_MAGIC) {
            return Err(OrtError("not a compiled NPU layer".into()));
        }
        let mut at = ARTIFACT_MAGIC.len();
        let u32_at = |at: &mut usize| {
            let v = u32::from_le_bytes(bytes[*at..*at + 4].try_into().expect("4 bytes"));
            *at += 4;
            v
        };
        let (k, n, max_tokens, inputs) = (
            u32_at(&mut at) as usize,
            u32_at(&mut at) as usize,
            u32_at(&mut at) as usize,
            u32_at(&mut at) as usize,
        );
        if inputs == 0 || inputs > 2 {
            return Err(OrtError(format!(
                "artifact declares {inputs} graph inputs, which this cannot bind"
            )));
        }
        let quant_at = |at: &mut usize| -> Result<Quant, OrtError> {
            let scale = f32::from_le_bytes(bytes[*at..*at + 4].try_into().expect("4 bytes"));
            *at += 4;
            let zero_point = u32_at(at);
            if !scale.is_finite() || scale <= 0.0 || zero_point > u8::MAX as u32 {
                return Err(OrtError("artifact has an unusable quantization".into()));
            }
            Ok(Quant {
                scale,
                zero_point: zero_point as u8,
            })
        };
        let x = quant_at(&mut at)?;
        let y = quant_at(&mut at)?;
        let smoothed = u32_at(&mut at) as usize;
        if smoothed != 0 && smoothed != k {
            return Err(OrtError(format!(
                "artifact carries {smoothed} channel scales for a {k}-feature input"
            )));
        }
        let mut unsmoothing = Vec::with_capacity(smoothed);
        for _ in 0..smoothed {
            let value = f32::from_le_bytes(
                bytes
                    .get(at..at + 4)
                    .ok_or_else(|| OrtError("artifact is truncated".into()))?
                    .try_into()
                    .expect("4 bytes"),
            );
            if !value.is_finite() || value <= 0.0 {
                return Err(OrtError("artifact has an unusable channel scale".into()));
            }
            unsmoothing.push(value);
            at += 4;
        }
        let len = u32_at(&mut at) as usize;

        let binary = bytes
            .get(at..at + len)
            .ok_or_else(|| OrtError("artifact is truncated".into()))?;
        if !binary.starts_with(b"\x7fELF") {
            return Err(OrtError("artifact does not hold an executable".into()));
        }
        if k == 0 || n == 0 || max_tokens == 0 {
            return Err(OrtError("artifact has a zero dimension".into()));
        }

        Ok(Self {
            binary: binary.to_vec(),
            k,
            n,
            max_tokens,
            inputs,
            x,
            y,
            unsmoothing,
        })
    }

    /// Loads this graph and gives up its host copy of the executable.
    ///
    /// The driver copies the binary during the load — verified by
    /// overwriting the source buffer immediately afterwards and getting
    /// byte-identical results — so keeping it costs memory and buys
    /// nothing. That matters at model scale rather than layer scale: an
    /// artifact embeds its own weights, so a whole model's artifacts come to
    /// roughly the size of the model. Reading them one at a time and calling
    /// this holds one at a time.
    ///
    /// Use [`Self::bind`] instead only to load the same artifact more than
    /// once, or to keep the bytes for [`Self::to_bytes`].
    pub fn into_bound(self, runtime: &crate::npu::NpuRuntime) -> Result<NpuLinear, OrtError> {
        let bound = self.bind(runtime)?;
        drop(self);
        Ok(bound)
    }

    /// Loads this graph into an NPU runtime, ready to run repeatedly.
    ///
    /// Keeps this artifact's copy of the executable; [`Self::into_bound`]
    /// releases it instead, which is what a serving path wants.
    ///
    /// Only safe in a process that has **not** opened [`NpuOrt`]. The two
    /// stacks corrupt each other's symbol bindings; the module doc explains
    /// what that looks like when it goes wrong, which is not always an
    /// error. Compile elsewhere, [`Self::from_bytes`] here, then bind.
    pub fn bind(&self, runtime: &crate::npu::NpuRuntime) -> Result<NpuLinear, OrtError> {
        let graph = runtime
            .load_graph_bytes(&self.binary)
            .map_err(|e| OrtError(format!("loading the compiled graph: {e}")))?;

        // The descriptors are the device's own account of what it will read
        // and write. Checking them against what was compiled turns a layout
        // misunderstanding into an error here rather than a wrong answer
        // several thousand inferences later.
        let expected_in = self.max_tokens * self.k;
        let expected_out = self.max_tokens * self.n;
        // `self.inputs` of them, all the same size: a two-input graph binds
        // the same activations twice (see [`CompiledLinear::inputs`]), so
        // "how many" is part of what was compiled rather than always one.
        match (graph.inputs(), graph.outputs()) {
            (inputs, [output])
                if inputs.len() == self.inputs
                    && inputs.iter().all(|i| i.size as usize == expected_in)
                    && output.size as usize == expected_out => {}
            (inputs, outputs) => {
                return Err(OrtError(format!(
                    "compiled graph has {} input(s) of {:?} bytes and {} output(s) of {:?} bytes, \
                     expected {} input(s) of {expected_in} bytes and one {expected_out}-byte \
                     output",
                    inputs.len(),
                    inputs.iter().map(|t| t.size).collect::<Vec<_>>(),
                    outputs.len(),
                    outputs.iter().map(|t| t.size).collect::<Vec<_>>(),
                    self.inputs,
                )));
            }
        }

        Ok(NpuLinear {
            graph,
            scratch: std::cell::RefCell::new(Scratch {
                // Sized unconditionally now: a graph with no smoothing
                // still writes here whenever `row_gain` scales a row.
                smoothed: vec![0.0; self.max_tokens * self.k],
                input: vec![0u8; self.max_tokens * self.k],
                outputs: Vec::new(),
                gain: vec![1.0; self.max_tokens],
            }),
            k: self.k,
            n: self.n,
            max_tokens: self.max_tokens,
            inputs: self.inputs,
            x: self.x,
            y: self.y,
            unsmoothing: self.unsmoothing.clone(),
        })
    }
}

/// A compiled linear projection loaded onto the NPU, ready to run.
///
/// Unlike a Zhouyi ONNX Runtime session, this runs as often as you ask it
/// to: the underlying [`crate::npu::NpuGraph`] reuses one job across calls.
pub struct NpuLinear {
    graph: crate::npu::NpuGraph,
    /// Buffers reused across calls, so a serving loop allocates nothing.
    ///
    /// Both are exactly one inference's worth and never grow. `RefCell`
    /// rather than `&mut self`: running a layer is logically a read, and the
    /// type is single-threaded already — it holds an
    /// [`std::rc::Rc`]-flavoured graph — so there is no sharing to guard.
    scratch: std::cell::RefCell<Scratch>,
    k: usize,
    n: usize,
    max_tokens: usize,
    /// See [`CompiledLinear::inputs`]. The same buffer is bound this many
    /// times.
    inputs: usize,
    x: Quant,
    y: Quant,
    /// See [`CompiledLinear::unsmoothing`]. Applied to `x` before it is
    /// quantized, because the weights on the device already carry the
    /// matching factor.
    unsmoothing: Vec<f32>,
}

/// [`NpuLinear`]'s reusable working memory.
#[derive(Default)]
struct Scratch {
    /// The activation after the per-channel smoothing, when the graph was
    /// compiled with any. Kept here so a steady-state serving loop still
    /// allocates nothing.
    smoothed: Vec<f32>,
    /// The quantized, transposed activation handed to the device.
    input: Vec<u8>,
    /// What the device wrote back, still in its own layout.
    outputs: Vec<Vec<u8>>,
    /// One [`row_gain`] per token, kept from the pass that scales the input
    /// up so the pass that scales the output back down can undo exactly the
    /// same factor. Sized by the token count, so a steady-state loop
    /// allocates it once.
    gain: Vec<f32>,
}

impl NpuLinear {
    pub fn in_features(&self) -> usize {
        self.k
    }

    pub fn out_features(&self) -> usize {
        self.n
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// The output quantization step — the smallest difference this layer can
    /// represent, and therefore the floor on its accuracy.
    ///
    /// Worth reading before judging a result. The output range is derived as
    /// a *worst case*: every product is assumed to reach its extreme and to
    /// accumulate in the same direction. Real activations cancel, so the
    /// true range is much narrower and this step is correspondingly coarser
    /// than it needs to be. Calibrating on real data and passing a measured
    /// range would tighten it; the bound is chosen because it can never
    /// saturate, which is the failure that silently destroys a result.
    pub fn output_scale(&self) -> f32 {
        self.y.scale
    }

    /// The activation quantization step, for the same reason.
    pub fn input_scale(&self) -> f32 {
        self.x.scale
    }

    /// Runs `y = x * W` on the NPU.
    ///
    /// `x` is `[tokens][k]` row-major and the result is `[tokens][n]`. The
    /// activation is quantized on the way in and the result dequantized on
    /// the way out, so the caller works in `f32` throughout and the
    /// integer-only nature of the device stays inside this function.
    pub fn forward(&self, x: &[f32], tokens: usize) -> Result<Vec<f32>, OrtError> {
        let mut out = Vec::new();
        self.forward_into(x, tokens, &mut out)?;
        Ok(out)
    }

    /// [`Self::forward`], writing into the caller's buffer.
    ///
    /// `out` is resized to `tokens * out_features()`, keeping whatever it
    /// already holds. With this and the scratch buffers a steady-state
    /// serving loop performs no allocation at all.
    pub fn forward_into(
        &self,
        x: &[f32],
        tokens: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), OrtError> {
        self.dispatch(x, tokens)?;
        self.collect(tokens, out)
    }

    /// Quantizes `x` and **dispatches** this projection without waiting.
    ///
    /// Split out from [`Self::forward_into`] so that two projections with no
    /// dependency between them can be in flight together — see
    /// [`NpuGatedFfn::forward_into`], where `gate` and `up` read the same
    /// activations and neither reads the other. Every `dispatch` must be
    /// matched by exactly one [`Self::collect`]: the scratch holding the
    /// quantized input and the per-row gains belongs to this projection and
    /// is overwritten by the next dispatch on it.
    pub fn dispatch(&self, x: &[f32], tokens: usize) -> Result<(), OrtError> {
        if tokens != self.max_tokens {
            return Err(OrtError(format!(
                "graph was compiled for {} tokens, {tokens} supplied",
                self.max_tokens
            )));
        }
        if x.len() != tokens * self.k {
            return Err(OrtError(format!(
                "input is {} values, expected tokens*k = {}",
                x.len(),
                tokens * self.k
            )));
        }

        let scratch = &mut *self.scratch.borrow_mut();

        // **What each row has to be multiplied by to fill the range this
        // graph was calibrated for**, measured on the smoothed values since
        // those are what gets quantized. Computed before anything is
        // written so the scaling pass below can fold it into the same
        // traversal that applies the smoothing.
        let (q_lo, q_hi) = if row_gain_enabled() {
            self.x.range()
        } else {
            // A range of zero width makes every gain 1.0, which is the
            // control arm: see `row_gain_enabled`.
            (0.0, 0.0)
        };
        scratch.gain.clear();
        scratch.gain.reserve(tokens);
        for row in x.chunks_exact(self.k) {
            let (mut lo, mut hi) = (0.0f32, 0.0f32);
            if self.unsmoothing.is_empty() {
                for v in row {
                    lo = lo.min(*v);
                    hi = hi.max(*v);
                }
            } else {
                for (v, s) in row.iter().zip(&self.unsmoothing) {
                    let v = v * s;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            scratch.gain.push(row_gain(lo, hi, q_lo, q_hi));
        }

        // **Smoothed first, if this graph was compiled that way.** The
        // weights already carry the matching factor, so skipping this would
        // not be a small error — it would be the whole per-channel scale
        // applied once instead of cancelling. See
        // `NpuOrt::smoothing_scales`.
        //
        // The per-row gain rides along in the same multiply: neither pass
        // is worth its own traversal of the activation.
        let rescaled = scratch.gain.iter().any(|g| *g != 1.0);
        let x = if self.unsmoothing.is_empty() && !rescaled {
            x
        } else {
            scratch.smoothed.resize(x.len(), 0.0);
            let rows = x
                .chunks_exact(self.k)
                .zip(scratch.smoothed.chunks_exact_mut(self.k))
                .zip(&scratch.gain);
            for ((row, out), gain) in rows {
                if self.unsmoothing.is_empty() {
                    for (v, o) in row.iter().zip(out) {
                        *o = v * gain;
                    }
                } else {
                    for ((v, s), o) in row.iter().zip(&self.unsmoothing).zip(out) {
                        *o = v * s * gain;
                    }
                }
            }
            &scratch.smoothed
        };

        scratch.input.resize(tokens * self.k, 0);
        quantize_transpose(x, tokens, self.k, self.x, &mut scratch.input);

        self.graph
            .start(
                // The same activations, bound once per declared input. Two
                // inputs is how the graph keeps the provider from merging
                // the gate's two projections back together — see
                // `GateActivation::GeluSigmoidTwoInput`.
                &vec![&scratch.input[..]; self.inputs],
                INFERENCE_TIMEOUT_MS,
            )
            .map_err(|e| OrtError(format!("dispatching to the NPU: {e}")))?;
        Ok(())
    }

    /// Waits for the job [`Self::dispatch`] started and writes its result
    /// into `out`, undoing the per-row gain on the way.
    ///
    /// `tokens` must be the same count `dispatch` was given.
    pub fn collect(&self, tokens: usize, out: &mut Vec<f32>) -> Result<(), OrtError> {
        let scratch = &mut *self.scratch.borrow_mut();
        self.graph
            .finish(&mut scratch.outputs, INFERENCE_TIMEOUT_MS)
            .map_err(|e| OrtError(format!("running on the NPU: {e}")))?;
        let raw = scratch
            .outputs
            .first()
            .ok_or_else(|| OrtError("the NPU returned no output tensor".into()))?;

        // Where a projection loses its signal, when a measurement asks.
        // Levels used says whether the input survived quantization at all;
        // the clipped fractions say whether either end of a static range
        // was too small for what this row actually produced.
        if gain_trace() {
            let levels = {
                let mut seen = [false; 256];
                for q in &scratch.input {
                    seen[*q as usize] = true;
                }
                seen.iter().filter(|s| **s).count()
            };
            let clipped = |v: &[u8]| {
                let n = v.iter().filter(|q| **q == 0 || **q == 255).count();
                100.0 * n as f32 / v.len().max(1) as f32
            };
            let out_levels = {
                let mut seen = [false; 256];
                for q in raw {
                    seen[*q as usize] = true;
                }
                seen.iter().filter(|s| **s).count()
            };
            eprintln!(
                "[npu gain] k={} n={} gain {:.2} in: {levels} levels, {:.1}% clipped; \
                 out: {out_levels} levels, {:.1}% clipped",
                self.k,
                self.n,
                scratch.gain.first().copied().unwrap_or(1.0),
                clipped(&scratch.input),
                clipped(raw),
            );
        }

        out.clear();
        out.resize(tokens * self.n, 0.0);
        dequantize_transpose(raw, tokens, self.n, self.y, out);
        // And back down. A projection is linear, so this cancels exactly
        // what the input pass applied and the caller sees the same function
        // it always did — only computed on a row the quantizer could
        // actually resolve.
        // The per-row test is the only guard: a row left at 1.0 skips the
        // multiply, and `dispatch` and `collect` no longer share a local to
        // carry a summary of that in.
        for (row, gain) in out.chunks_exact_mut(self.n).zip(&scratch.gain) {
            if *gain != 1.0 {
                let undo = 1.0 / gain;
                for v in row {
                    *v *= undo;
                }
            }
        }
        Ok(())
    }
}

/// Quantizes `[tokens][k]` row-major `f32` into the `[k][tokens]` `uint8`
/// layout the convolution reads, in one pass.
///
/// The naive form of this — a scalar loop writing `out[c * tokens + t]` —
/// touches a different cache line on every store and divides once per
/// element. At prefill shapes it cost more than the inference it was
/// feeding: 512x128x512 measured 1.4 ms end to end against 0.29 ms on the
/// device itself. Hence tiling, and hence the vector path below, which took
/// the same shape to 0.69 ms.
fn quantize_transpose(x: &[f32], tokens: usize, k: usize, q: Quant, out: &mut [u8]) {
    debug_assert_eq!(x.len(), tokens * k);
    debug_assert_eq!(out.len(), tokens * k);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64, and the lengths are asserted
        // above — the helper reads and writes only within them.
        unsafe { quantize_transpose_neon(x, tokens, k, q, out) };
        return;
    }
    #[allow(unreachable_code)]
    quantize_transpose_scalar(x, tokens, k, q, out);
}

/// The definition [`quantize_transpose`] must match, and the fallback on any
/// architecture without a vector path.
///
/// Blocked over the feature axis even here: the destination column for one
/// block stays inside L1 across every token, which is most of the win and
/// costs nothing in clarity.
fn quantize_transpose_scalar(x: &[f32], tokens: usize, k: usize, q: Quant, out: &mut [u8]) {
    const BLOCK: usize = 64;
    for c0 in (0..k).step_by(BLOCK) {
        let c1 = (c0 + BLOCK).min(k);
        for t in 0..tokens {
            let row = &x[t * k + c0..t * k + c1];
            for (offset, value) in row.iter().enumerate() {
                out[(c0 + offset) * tokens + t] = q.quantize(*value);
            }
        }
    }
}

/// Dequantizes the `[n][tokens]` `uint8` the device returns back into
/// `[tokens][n]` row-major `f32`.
fn dequantize_transpose(raw: &[u8], tokens: usize, n: usize, q: Quant, out: &mut [f32]) {
    debug_assert_eq!(raw.len(), tokens * n);
    debug_assert_eq!(out.len(), tokens * n);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: as above.
        unsafe { dequantize_transpose_neon(raw, tokens, n, q, out) };
        return;
    }
    #[allow(unreachable_code)]
    dequantize_transpose_scalar(raw, tokens, n, q, out);
}

/// The definition [`dequantize_transpose`] must match.
fn dequantize_transpose_scalar(raw: &[u8], tokens: usize, n: usize, q: Quant, out: &mut [f32]) {
    const BLOCK: usize = 64;
    for f0 in (0..n).step_by(BLOCK) {
        let f1 = (f0 + BLOCK).min(n);
        for t in 0..tokens {
            for f in f0..f1 {
                out[t * n + f] = q.dequantize(raw[f * tokens + t]);
            }
        }
    }
}

/// Transposes a 16x16 block of bytes held one row per vector register.
///
/// Four stages of `trn1`/`trn2`, each doubling the width being exchanged —
/// bytes, then pairs, then quads, then halves of the register. 16 rows in,
/// the same 16 registers holding the transpose out, with no memory traffic
/// in between. This is the standard recursive vector transpose; it is
/// checked against a scalar transpose in the tests rather than trusted.
///
/// # Safety
/// Requires NEON, which is baseline on aarch64.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_16x16_u8(rows: &mut [std::arch::aarch64::uint8x16_t; 16]) {
    use std::arch::aarch64::*;
    unsafe {
        for i in (0..16).step_by(2) {
            let (a, b) = (rows[i], rows[i + 1]);
            rows[i] = vtrn1q_u8(a, b);
            rows[i + 1] = vtrn2q_u8(a, b);
        }
        for base in (0..16).step_by(4) {
            for j in 0..2 {
                let a = vreinterpretq_u16_u8(rows[base + j]);
                let b = vreinterpretq_u16_u8(rows[base + j + 2]);
                rows[base + j] = vreinterpretq_u8_u16(vtrn1q_u16(a, b));
                rows[base + j + 2] = vreinterpretq_u8_u16(vtrn2q_u16(a, b));
            }
        }
        for base in (0..16).step_by(8) {
            for j in 0..4 {
                let a = vreinterpretq_u32_u8(rows[base + j]);
                let b = vreinterpretq_u32_u8(rows[base + j + 4]);
                rows[base + j] = vreinterpretq_u8_u32(vtrn1q_u32(a, b));
                rows[base + j + 4] = vreinterpretq_u8_u32(vtrn2q_u32(a, b));
            }
        }
        for j in 0..8 {
            let a = vreinterpretq_u64_u8(rows[j]);
            let b = vreinterpretq_u64_u8(rows[j + 8]);
            rows[j] = vreinterpretq_u8_u64(vtrn1q_u64(a, b));
            rows[j + 8] = vreinterpretq_u8_u64(vtrn2q_u64(a, b));
        }
    }
}

/// Quantizes 16 contiguous `f32` into 16 `uint8`.
///
/// Divides rather than multiplying by a reciprocal, so that this agrees with
/// [`Quant::quantize`] bit for bit — a reciprocal differs in the last place
/// often enough to flip a rounding tie, and a vector path that disagrees
/// with its own scalar definition is worse than no vector path. aarch64 has
/// a real `fdiv`, so this costs little.
///
/// `fcvtas` rounds halfway cases away from zero, which is what `f32::round`
/// does; the saturating narrows reproduce the `clamp(0, 255)` exactly, and
/// both forms turn NaN into zero.
///
/// # Safety
/// `src` must have 16 readable `f32`. Requires NEON.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn quantize16(
    src: *const f32,
    scale: std::arch::aarch64::float32x4_t,
    zero: std::arch::aarch64::int32x4_t,
) -> std::arch::aarch64::uint8x16_t {
    use std::arch::aarch64::*;
    unsafe {
        let mut lanes = [vdupq_n_s32(0); 4];
        for (i, lane) in lanes.iter_mut().enumerate() {
            let v = vld1q_f32(src.add(i * 4));
            *lane = vaddq_s32(vcvtaq_s32_f32(vdivq_f32(v, scale)), zero);
        }
        let lo = vcombine_s16(vqmovn_s32(lanes[0]), vqmovn_s32(lanes[1]));
        let hi = vcombine_s16(vqmovn_s32(lanes[2]), vqmovn_s32(lanes[3]));
        vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi))
    }
}

/// [`quantize_transpose`]'s vector path: quantize a 16x16 tile into
/// registers, transpose it there, and store 16 contiguous bytes per feature.
///
/// Every load and every store is a full vector; the transpose that used to
/// be 256 scattered byte stores is four register stages. Edges fall back to
/// the scalar definition, which keeps the two in step by construction.
///
/// # Safety
/// Requires NEON. `x` and `out` must both hold `tokens * k` elements.
#[cfg(target_arch = "aarch64")]
unsafe fn quantize_transpose_neon(x: &[f32], tokens: usize, k: usize, q: Quant, out: &mut [u8]) {
    use std::arch::aarch64::*;
    unsafe {
        let scale = vdupq_n_f32(q.scale);
        let zero = vdupq_n_s32(q.zero_point as i32);
        let (tile_t, tile_c) = (tokens / 16 * 16, k / 16 * 16);

        for c0 in (0..tile_c).step_by(16) {
            for t0 in (0..tile_t).step_by(16) {
                let mut rows = [vdupq_n_u8(0); 16];
                for (r, row) in rows.iter_mut().enumerate() {
                    *row = quantize16(x.as_ptr().add((t0 + r) * k + c0), scale, zero);
                }
                transpose_16x16_u8(&mut rows);
                for (r, row) in rows.iter().enumerate() {
                    vst1q_u8(out.as_mut_ptr().add((c0 + r) * tokens + t0), *row);
                }
            }
        }

        // The right and bottom edges, where a whole tile does not fit.
        if tile_c < k {
            quantize_transpose_edge(x, tokens, k, q, out, tile_c..k, 0..tokens);
        }
        if tile_t < tokens {
            quantize_transpose_edge(x, tokens, k, q, out, 0..tile_c, tile_t..tokens);
        }
    }
}

/// The scalar definition applied to one rectangle of the problem, for the
/// edges a 16x16 tiling cannot cover.
#[cfg(target_arch = "aarch64")]
fn quantize_transpose_edge(
    x: &[f32],
    tokens: usize,
    k: usize,
    q: Quant,
    out: &mut [u8],
    features: std::ops::Range<usize>,
    token_range: std::ops::Range<usize>,
) {
    for c in features {
        for t in token_range.clone() {
            out[c * tokens + t] = q.quantize(x[t * k + c]);
        }
    }
}

/// [`dequantize_transpose`]'s vector path.
///
/// The transpose happens in the byte domain, where a register holds 16
/// values rather than 4, and only then are the bytes widened to `f32` —
/// along the output's contiguous axis, so the stores are vectors too.
///
/// # Safety
/// Requires NEON. `raw` and `out` must both hold `tokens * n` elements.
#[cfg(target_arch = "aarch64")]
unsafe fn dequantize_transpose_neon(
    raw: &[u8],
    tokens: usize,
    n: usize,
    q: Quant,
    out: &mut [f32],
) {
    use std::arch::aarch64::*;
    unsafe {
        let scale = vdupq_n_f32(q.scale);
        let zero = vdupq_n_s32(q.zero_point as i32);
        let (tile_t, tile_f) = (tokens / 16 * 16, n / 16 * 16);

        for f0 in (0..tile_f).step_by(16) {
            for t0 in (0..tile_t).step_by(16) {
                let mut rows = [vdupq_n_u8(0); 16];
                for (r, row) in rows.iter_mut().enumerate() {
                    *row = vld1q_u8(raw.as_ptr().add((f0 + r) * tokens + t0));
                }
                // Rows are features, columns are tokens; transposing gives
                // one token per row, which is what the output wants.
                transpose_16x16_u8(&mut rows);
                for (r, row) in rows.iter().enumerate() {
                    let wide = [vmovl_u8(vget_low_u8(*row)), vmovl_u8(vget_high_u8(*row))];
                    let base = out.as_mut_ptr().add((t0 + r) * n + f0);
                    for (half, w) in wide.iter().enumerate() {
                        let parts = [vget_low_u16(*w), vget_high_u16(*w)];
                        for (quarter, part) in parts.iter().enumerate() {
                            let as_i32 = vreinterpretq_s32_u32(vmovl_u16(*part));
                            let centred = vsubq_s32(as_i32, zero);
                            let value = vmulq_f32(vcvtq_f32_s32(centred), scale);
                            vst1q_f32(base.add(half * 8 + quarter * 4), value);
                        }
                    }
                }
            }
        }

        if tile_f < n {
            dequantize_transpose_edge(raw, tokens, n, q, out, tile_f..n, 0..tokens);
        }
        if tile_t < tokens {
            dequantize_transpose_edge(raw, tokens, n, q, out, 0..tile_f, tile_t..tokens);
        }
    }
}

/// The scalar definition applied to one rectangle, for the edges.
#[cfg(target_arch = "aarch64")]
fn dequantize_transpose_edge(
    raw: &[u8],
    tokens: usize,
    n: usize,
    q: Quant,
    out: &mut [f32],
    features: std::ops::Range<usize>,
    token_range: std::ops::Range<usize>,
) {
    for f in features {
        for t in token_range.clone() {
            out[t * n + f] = q.dequantize(raw[f * tokens + t]);
        }
    }
}

/// How long to wait for one inference before giving up.
///
/// Generous by design: a projection takes well under a millisecond, so this
/// is not a deadline but a guard against a job that will never complete —
/// which is a failure this stack is known to produce.
const INFERENCE_TIMEOUT_MS: i32 = 10_000;

#[cfg(test)]
mod tests {
    use super::*;

    /// Quantization has to survive a round trip to within half a step, or
    /// every result computed through it is wrong by more than the hardware's
    /// own precision.
    #[test]
    fn quantization_round_trips_within_half_a_step() {
        let q = Quant::covering(-2.0, 6.0);
        for v in [-2.0f32, -0.5, 0.0, 1.25, 6.0] {
            let back = q.dequantize(q.quantize(v));
            assert!(
                (back - v).abs() <= q.scale,
                "{v} came back as {back}, scale {}",
                q.scale
            );
        }
    }

    /// A range that never moves still needs a usable scale rather than a
    /// division by zero.
    #[test]
    fn a_degenerate_range_still_yields_a_usable_scale() {
        let q = Quant::covering(3.0, 3.0);
        assert!(q.scale > 0.0);
        assert!(q.dequantize(q.quantize(3.0)).is_finite());
    }

    /// The serializer must produce something that at least carries ONNX's
    /// own framing: an `ir_version` field, and the operator-set and graph
    /// submessages. A malformed model is rejected far away from here, with
    /// a message that doesn't point back at this function.
    #[test]
    fn the_generated_model_has_onnx_framing() {
        let w = vec![0u8; 4 * 3];
        let q = Quant::covering(0.0, 1.0);
        let model = build_stack_model(&[3, 4], 2, &[&w], &[q, q], &[q]);
        assert!(model.len() > 32, "model implausibly small: {}", model.len());
        // field 1 (ir_version), varint -> tag byte 0x08.
        assert_eq!(model[0], 0x08, "model does not start with ir_version");
        // The weights must appear verbatim as raw_data.
        assert!(
            model.windows(w.len()).any(|c| c == w.as_slice()),
            "weight bytes are not present in the model"
        );
    }

    /// Opening the compiler answers rather than failing, on any machine.
    ///
    /// Ignored for the reason the module doc gives, and this is where that
    /// rule bites hardest: loading the provider's libraries is enough to
    /// capture `libnoe`'s symbol bindings, so a default `cargo test` run
    /// that touched both modules segfaulted in `libnoe`'s static destructor
    /// at process exit — after every test had passed. Nothing in the default
    /// run may open this stack while [`crate::npu`]'s tests are also in it.
    #[test]
    #[ignore = "loads the provider's libraries; conflicts with crate::npu in one process"]
    fn opening_the_compiler_answers_rather_than_failing() {
        let _ = NpuOrt::open();
    }

    /// A weight slice of the wrong length is caught here, before anything
    /// is handed to the provider.
    ///
    /// Ignored for the same process-conflict reason as the test above.
    #[test]
    #[ignore = "loads the provider's libraries; conflicts with crate::npu in one process"]
    fn mismatched_weights_are_rejected() {
        let Some(ort) = NpuOrt::open() else {
            return; // No NPU stack on this machine.
        };
        let err = ort.compile_linear(&[0.0; 5], 3, 4, 2, (0.0, 1.0));
        assert!(
            err.is_err(),
            "a 5-value weight matrix for 3x4 must be rejected"
        );
    }

    /// Builds a minimal EPContext model around a payload and reads it back.
    #[test]
    fn a_context_binary_is_recovered_from_its_attribute() {
        let payload = b"\x7fELFsome compiled graph".to_vec();
        let mut model = b"...noise...ep_cache_context".to_vec();
        model.push(0x22);
        let mut len = payload.len();
        loop {
            let mut byte = (len & 0x7f) as u8;
            len >>= 7;
            if len > 0 {
                byte |= 0x80;
            }
            model.push(byte);
            if len == 0 {
                break;
            }
        }
        model.extend_from_slice(&payload);
        model.extend_from_slice(b"trailing bytes that are not the payload");

        assert_eq!(extract_context_binary(&model).expect("extract"), payload);
    }

    /// A model with no such attribute, a truncated one, and one holding
    /// something that is not an executable are all errors rather than
    /// silently wrong bytes.
    #[test]
    fn a_malformed_context_model_is_an_error() {
        assert!(extract_context_binary(b"no attribute here").is_err());

        let mut truncated = b"ep_cache_context".to_vec();
        truncated.extend_from_slice(&[0x22, 0x40]); // claims 64 bytes, has none
        assert!(extract_context_binary(&truncated).is_err());

        let mut not_elf = b"ep_cache_context".to_vec();
        not_elf.extend_from_slice(&[0x22, 0x04]);
        not_elf.extend_from_slice(b"/tmp");
        assert!(
            extract_context_binary(&not_elf).is_err(),
            "a path instead of an executable means embed mode did not apply"
        );
    }

    /// A small deterministic generator, so a failure is reproducible.
    fn pseudo_random(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    /// **The vector path must be the scalar path.**
    ///
    /// Written because [`quantize_transpose_neon`] is four stages of
    /// register shuffling that no amount of reading proves correct, and
    /// because it runs on every inference: a transpose that is wrong in one
    /// lane would show up as a slightly wrong answer, which is the hardest
    /// kind of bug to notice. Shapes deliberately include ones that are not
    /// multiples of 16 on either axis, since those take the edge paths.
    #[test]
    fn quantize_transpose_agrees_with_its_scalar_definition() {
        let quant = Quant::covering(-1.5, 2.5);
        let mut seed = 0x5eed;
        for (tokens, k) in [
            (16, 16),
            (32, 64),
            (8, 8),
            (1, 1),
            (17, 33),
            (15, 16),
            (16, 15),
            (7, 129),
            (128, 512),
        ] {
            // Values reaching past the quantized range on both sides, so the
            // saturating narrows are exercised rather than assumed.
            let x: Vec<f32> = (0..tokens * k)
                .map(|_| pseudo_random(&mut seed) * 4.0)
                .collect();

            let mut want = vec![0u8; tokens * k];
            quantize_transpose_scalar(&x, tokens, k, quant, &mut want);
            let mut got = vec![0u8; tokens * k];
            quantize_transpose(&x, tokens, k, quant, &mut got);

            assert_eq!(got, want, "quantize_transpose differs at {tokens}x{k}");

            // And it really is a transpose, not merely self-consistent.
            for t in 0..tokens {
                for c in 0..k {
                    assert_eq!(got[c * tokens + t], quant.quantize(x[t * k + c]));
                }
            }
        }
    }

    /// The same, for the way back.
    #[test]
    fn dequantize_transpose_agrees_with_its_scalar_definition() {
        let quant = Quant::covering(-8.0, 8.0);
        let mut seed = 0xd0d0;
        for (tokens, n) in [
            (16, 16),
            (32, 64),
            (8, 8),
            (1, 1),
            (17, 33),
            (15, 16),
            (16, 15),
            (7, 129),
            (128, 512),
        ] {
            let raw: Vec<u8> = (0..tokens * n)
                .map(|_| ((pseudo_random(&mut seed) + 1.0) * 127.5) as u8)
                .collect();

            let mut want = vec![0.0f32; tokens * n];
            dequantize_transpose_scalar(&raw, tokens, n, quant, &mut want);
            let mut got = vec![0.0f32; tokens * n];
            dequantize_transpose(&raw, tokens, n, quant, &mut got);

            assert_eq!(got, want, "dequantize_transpose differs at {tokens}x{n}");

            for t in 0..tokens {
                for f in 0..n {
                    assert_eq!(got[t * n + f], quant.dequantize(raw[f * tokens + t]));
                }
            }
        }
    }

    /// The register transpose on its own, against the obvious loop.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn the_register_transpose_is_a_transpose() {
        use std::arch::aarch64::*;
        let block: Vec<u8> = (0..256).map(|i| (i * 7 % 251) as u8).collect();

        // SAFETY: NEON is baseline on aarch64; every load and store is
        // 16 bytes inside a 256-byte block.
        let got = unsafe {
            let mut rows = [vdupq_n_u8(0); 16];
            for (r, row) in rows.iter_mut().enumerate() {
                *row = vld1q_u8(block.as_ptr().add(r * 16));
            }
            transpose_16x16_u8(&mut rows);
            let mut out = vec![0u8; 256];
            for (r, row) in rows.iter().enumerate() {
                vst1q_u8(out.as_mut_ptr().add(r * 16), *row);
            }
            out
        };

        for r in 0..16 {
            for c in 0..16 {
                assert_eq!(got[r * 16 + c], block[c * 16 + r], "at ({r}, {c})");
            }
        }
    }

    /// The fixture both halves of the hardware round trip share, so the
    /// serving side can recompute a reference for what the compiling side
    /// built without either half trusting the other.
    mod fixture {
        pub const K: usize = 64;
        pub const N: usize = 32;
        pub const TOKENS: usize = 8;

        /// Deliberately signed on both sides: a symmetric-only quantizer
        /// would pass a non-negative test and fail here.
        pub fn weights() -> Vec<f32> {
            (0..N * K).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect()
        }

        pub fn input() -> Vec<f32> {
            (0..TOKENS * K)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.25)
                .collect()
        }

        /// Where the compiling half leaves its artifact for the serving half.
        pub fn artifact_path() -> std::path::PathBuf {
            std::env::var_os("ORANGU_NPU_ARTIFACT")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::env::temp_dir().join("orangu-npu-fixture.npu"))
        }
    }

    /// An artifact survives a trip through bytes unchanged.
    #[test]
    fn a_compiled_layer_round_trips_through_bytes() {
        let original = CompiledLinear {
            binary: b"\x7fELF and then some payload".to_vec(),
            k: 64,
            n: 32,
            max_tokens: 8,
            inputs: 1,
            x: Quant::covering(-1.0, 1.0),
            y: Quant::covering(-8.0, 8.0),
            // Non-trivial, so the round trip has to carry them: an artifact
            // that lost these would quantize a smoothed input against
            // unsmoothed weights and be wrong by the whole scale.
            unsmoothing: (0..64).map(|i| 1.0 + i as f32 / 64.0).collect(),
        };
        let restored = CompiledLinear::from_bytes(&original.to_bytes()).expect("round trip");
        assert_eq!(restored.unsmoothing, original.unsmoothing);

        assert_eq!(restored.binary(), original.binary());
        assert_eq!(restored.in_features(), original.in_features());
        assert_eq!(restored.out_features(), original.out_features());
        assert_eq!(restored.max_tokens(), original.max_tokens());
        assert_eq!(restored.x.scale, original.x.scale);
        assert_eq!(restored.x.zero_point, original.x.zero_point);
        assert_eq!(restored.y.scale, original.y.scale);
        assert_eq!(restored.y.zero_point, original.y.zero_point);
    }

    /// Nothing that is not an artifact is accepted as one.
    #[test]
    fn a_damaged_artifact_is_rejected() {
        assert!(CompiledLinear::from_bytes(b"").is_err());
        assert!(CompiledLinear::from_bytes(b"ORANGU-NPU-LINEAR-9 whatever follows").is_err());

        let good = CompiledLinear {
            binary: b"\x7fELF payload".to_vec(),
            k: 4,
            n: 4,
            max_tokens: 1,
            inputs: 1,
            x: Quant::covering(-1.0, 1.0),
            y: Quant::covering(-1.0, 1.0),
            unsmoothing: Vec::new(),
        }
        .to_bytes();

        assert!(
            CompiledLinear::from_bytes(&good[..good.len() - 4]).is_err(),
            "a truncated payload must not be read as a short graph"
        );

        // An executable that is not one: the byte that says so, flipped.
        let mut not_elf = good.clone();
        let elf_at = not_elf.len() - b"\x7fELF payload".len();
        not_elf[elf_at] = b'M';
        assert!(CompiledLinear::from_bytes(&not_elf).is_err());
    }

    /// The compiling half of the hardware round trip: compile the fixture
    /// and leave the artifact where the serving half will find it.
    ///
    /// Ignored, and separate from the run below, because compiling and
    /// serving cannot share a process — see the module doc. Run the pair:
    ///
    /// ```text
    /// cargo test --lib npu_ort::tests::compiling -- --ignored --nocapture
    /// cargo test --lib npu_ort::tests::a_compiled_layer -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "hardware; must run in its own process — see the module doc"]
    fn compiling_a_linear_layer_produces_a_loadable_artifact() {
        use fixture::{K, N, TOKENS};
        let Some(ort) = NpuOrt::open() else {
            return; // No NPU compiler on this machine.
        };
        let compiled = match ort.compile_linear(&fixture::weights(), K, N, TOKENS, (-1.0, 1.0)) {
            Ok(l) => l,
            // A provider that refuses to compile is a finding, not a silent
            // pass: fail loudly on a machine that has the stack.
            Err(e) => panic!("compiling a linear layer failed: {e}"),
        };

        assert_eq!(compiled.in_features(), K);
        assert_eq!(compiled.out_features(), N);
        assert_eq!(compiled.max_tokens(), TOKENS);
        assert!(
            compiled.binary().starts_with(b"\x7fELF"),
            "the compiled artifact should be an AIPU executable"
        );

        let path = fixture::artifact_path();
        std::fs::write(&path, compiled.to_bytes()).expect("writing the artifact");
        println!(
            "wrote {} ({} bytes)",
            path.display(),
            compiled.binary().len()
        );
    }

    /// The serving half: load what was compiled, run it more than once, and
    /// check it against a plain CPU matmul.
    ///
    /// Needs the artifact the test above writes. It opens no compiler, which
    /// is what makes running here safe.
    #[test]
    #[ignore = "hardware; needs the artifact from compiling_a_linear_layer_produces_a_loadable_artifact"]
    fn a_compiled_layer_runs_on_the_npu_and_matches_a_cpu_reference() {
        use fixture::{K, N, TOKENS};
        let path = fixture::artifact_path();
        let Ok(bytes) = std::fs::read(&path) else {
            println!(
                "no artifact at {}; run the compiling test first",
                path.display()
            );
            return;
        };
        let compiled = CompiledLinear::from_bytes(&bytes).expect("reading the artifact");
        let Some(runtime) = crate::npu::NpuRuntime::open() else {
            return; // No NPU runtime on this machine.
        };
        let linear = compiled.bind(&runtime).expect("binding the compiled graph");

        let (weights, x) = (fixture::weights(), fixture::input());
        let got = linear.forward(&x, TOKENS).expect("forward failed");
        assert_eq!(got.len(), TOKENS * N);

        let mut want = vec![0.0f32; TOKENS * N];
        for t in 0..TOKENS {
            for o in 0..N {
                let mut acc = 0.0f32;
                for i in 0..K {
                    acc += x[t * K + i] * weights[o * K + i];
                }
                want[t * N + o] = acc;
            }
        }

        // The floor on accuracy is the device's own precision, not a number
        // picked here: one step of output quantization, plus the activation
        // error carried through the accumulation. Anything inside that is
        // the hardware behaving correctly; anything beyond means the mapping
        // is wrong.
        let tolerance = 2.0 * linear.output_scale() + (K as f32) * linear.input_scale() * 0.5;
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= tolerance,
            "NPU result differs by {worst}, tolerance {tolerance}"
        );
        assert!(
            got.iter().any(|v| *v != 0.0),
            "every output was zero, which is what a failed run looks like"
        );

        // The whole point of compiling to bytes and running through NOE: it
        // runs again, and agrees with itself. An ONNX Runtime session would
        // hang on this call rather than answer it.
        for _ in 0..20 {
            assert_eq!(
                linear.forward(&x, TOKENS).expect("repeat inference"),
                got,
                "the same input gave a different answer"
            );
        }

        // A graph is compiled for one token count; anything else must be an
        // error rather than a mis-shaped read.
        assert!(linear.forward(&x, TOKENS + 1).is_err());
    }

    /// Shapes that do not describe a stack are refused before the provider
    /// is involved.
    ///
    /// Ignored despite reaching no hardware: [`NpuOrt::open`] loads the
    /// provider's libraries, and that alone is enough to break
    /// [`crate::npu`] later in the same process.
    #[test]
    #[ignore = "loads the provider's libraries; conflicts with crate::npu in one process"]
    fn a_stack_with_inconsistent_shapes_is_rejected() {
        let Some(ort) = NpuOrt::open() else {
            return; // No NPU compiler on this machine.
        };
        let a = vec![0.0f32; 8 * 4];
        let b = vec![0.0f32; 4 * 2];

        let cal = vec![0.5f32; 4];

        // Three dimensions describe two layers, not one.
        assert!(ort.compile_stack(&[4, 8, 2], &[&a], 1, &cal).is_err());
        // A layer whose weight count does not match the dimensions around it.
        assert!(ort.compile_stack(&[4, 8, 4], &[&a, &b], 1, &cal).is_err());
        // Degenerate dimensions and token counts.
        assert!(ort.compile_stack(&[0, 8], &[&a], 1, &cal).is_err());
        assert!(ort.compile_stack(&[4, 8], &[&a], 0, &cal).is_err());
        assert!(ort.compile_stack(&[], &[], 1, &cal).is_err());
        // Calibration that is absent, or not whole rows of the input width.
        assert!(ort.compile_stack(&[4, 8], &[&a], 1, &[]).is_err());
        assert!(ort.compile_stack(&[4, 8], &[&a], 1, &[0.5; 6]).is_err());
    }

    /// The fused stack fixture: a two-layer MLP with a rectifier between.
    mod stack_fixture {
        pub const K: usize = 64;
        pub const H: usize = 128;
        pub const N: usize = 32;
        pub const TOKENS: usize = 8;

        pub fn first() -> Vec<f32> {
            (0..H * K).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect()
        }

        pub fn second() -> Vec<f32> {
            (0..N * H).map(|i| ((i % 7) as f32 - 3.0) * 0.08).collect()
        }

        pub fn input() -> Vec<f32> {
            (0..TOKENS * K)
                .map(|i| ((i % 5) as f32 - 2.0) * 0.3)
                .collect()
        }

        pub fn path() -> std::path::PathBuf {
            std::env::temp_dir().join("orangu-npu-stack.npu")
        }
    }

    /// Compiling half of the fused-stack round trip.
    #[test]
    #[ignore = "hardware; must run in its own process — see the module doc"]
    fn compiling_a_stack_produces_a_loadable_artifact() {
        use stack_fixture::{H, K, N, TOKENS};
        let Some(ort) = NpuOrt::open() else {
            return;
        };
        let (first, second) = (stack_fixture::first(), stack_fixture::second());
        // Calibrated on the very input the run half will use, which is the
        // best case for quantization and therefore the right way to test the
        // mechanism rather than the calibration set.
        let calibration = stack_fixture::input();
        let compiled = match ort.compile_stack(&[K, H, N], &[&first, &second], TOKENS, &calibration)
        {
            Ok(c) => c,
            Err(e) => panic!("compiling a two-layer stack failed: {e}"),
        };

        assert_eq!(
            compiled.in_features(),
            K,
            "the stack takes the first layer's input"
        );
        assert_eq!(
            compiled.out_features(),
            N,
            "and returns the last layer's output"
        );
        assert!(compiled.binary().starts_with(b"\x7fELF"));

        std::fs::write(stack_fixture::path(), compiled.to_bytes()).expect("writing the artifact");
        println!(
            "wrote {} ({} bytes)",
            stack_fixture::path().display(),
            compiled.binary().len()
        );
    }

    /// **Two layers in one graph compute what two layers compute.**
    ///
    /// The point of fusing is that the intermediate never leaves the device,
    /// so nothing on the host ever sees it — which is exactly what makes a
    /// mistake here invisible without this test.
    #[test]
    #[ignore = "hardware; needs the artifact from compiling_a_stack_produces_a_loadable_artifact"]
    fn a_compiled_stack_matches_a_cpu_reference() {
        use stack_fixture::{H, K, N, TOKENS};
        let Ok(bytes) = std::fs::read(stack_fixture::path()) else {
            println!("no stack artifact; run the compiling test first");
            return;
        };
        let compiled = CompiledLinear::from_bytes(&bytes).expect("reading the artifact");
        let Some(runtime) = crate::npu::NpuRuntime::open() else {
            return;
        };
        let layer = compiled.into_bound(&runtime).expect("binding the stack");

        let (first, second, x) = (
            stack_fixture::first(),
            stack_fixture::second(),
            stack_fixture::input(),
        );
        let got = layer.forward(&x, TOKENS).expect("forward failed");
        assert_eq!(got.len(), TOKENS * N);

        let mut hidden = vec![0.0f32; TOKENS * H];
        for t in 0..TOKENS {
            for o in 0..H {
                let mut acc = 0.0f32;
                for i in 0..K {
                    acc += x[t * K + i] * first[o * K + i];
                }
                hidden[t * H + o] = acc.max(0.0);
            }
        }
        let mut want = vec![0.0f32; TOKENS * N];
        for t in 0..TOKENS {
            for o in 0..N {
                let mut acc = 0.0f32;
                for i in 0..H {
                    acc += hidden[t * H + i] * second[o * H + i];
                }
                want[t * N + o] = acc;
            }
        }

        // Two quantized layers, so two layers' worth of step to allow for:
        // the second layer's own output step, plus the first layer's error
        // carried through the second layer's accumulation.
        let tolerance = 2.0 * layer.output_scale() + (H as f32) * layer.input_scale() * 0.5;
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= tolerance,
            "fused stack differs by {worst}, tolerance {tolerance}"
        );
        assert!(got.iter().any(|v| *v != 0.0), "every output was zero");

        // And it repeats, like any other bound graph.
        assert_eq!(layer.forward(&x, TOKENS).expect("repeat"), got);
    }

    /// A real projection out of a real model, rather than a synthetic one.
    ///
    /// Points at a GGUF through `ORANGU_GGUF`, defaulting to the multimodal
    /// projector that ships beside Gemma 4. That file is the best-shaped work
    /// this device could be given: a vision encoder runs a *fixed* 196
    /// patches every time, so one compiled graph serves forever, where a
    /// language model would need one per token count. It also ships an
    /// activation range beside every weight — `input_min`, `input_max` —
    /// which is exactly the calibration a static quantizer needs and
    /// otherwise has to guess.
    mod real_model {
        /// 14x14 patches of a 224x224 image at patch size 16 — the vision
        /// encoder's fixed token count, and the default here because a fixed
        /// count is what makes one compiled graph last.
        pub const PATCHES: usize = 196;

        pub fn gguf() -> Option<std::path::PathBuf> {
            let path = std::path::PathBuf::from(std::env::var_os("ORANGU_GGUF")?);
            path.is_file().then_some(path)
        }

        /// Which projection to compile. Overridable so the same test can be
        /// pointed at either model in the pair — the projector's vision
        /// blocks, or the language model's own.
        pub fn tensor() -> String {
            std::env::var("ORANGU_NPU_TENSOR").unwrap_or_else(|_| "v.blk.0.ffn_down.weight".into())
        }

        pub fn tokens() -> usize {
            std::env::var("ORANGU_NPU_TOKENS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(PATCHES)
        }

        pub fn artifact() -> std::path::PathBuf {
            std::env::temp_dir().join("orangu-npu-real.npu")
        }

        /// One projection, as the file describes it.
        pub struct Layer {
            pub weights: Vec<f32>,
            pub in_features: usize,
            pub out_features: usize,
            /// The activation range the model was calibrated with.
            pub range: (f32, f32),
        }

        /// The weights, and the activation range the model was calibrated
        /// with — falling back to a symmetric guess if this file carries no
        /// range for the tensor.
        pub fn layer(path: &std::path::Path) -> anyhow::Result<Layer> {
            let name = tensor();
            let gguf = crate::gguf::GgufFile::open(path)?;
            let info = gguf
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| anyhow::anyhow!("{name} is not in {}", path.display()))?;
            let (k, n) = (info.dims[0] as usize, info.dims[1] as usize);
            let weights = gguf.read_tensor(path, &name)?;

            let stem = name.trim_end_matches(".weight");
            let range = match (
                gguf.read_scalar(path, &format!("{stem}.input_min")),
                gguf.read_scalar(path, &format!("{stem}.input_max")),
            ) {
                (Some(lo), Some(hi)) if hi > lo => (lo, hi),
                _ => (-1.0, 1.0),
            };
            Ok(Layer {
                weights,
                in_features: k,
                out_features: n,
                range,
            })
        }
    }

    /// A whole feed-forward block, read out of a real model.
    mod real_ffn {
        /// The block to take, as a tensor-name prefix. The projector's
        /// vision blocks are the default: they are the right size to compile
        /// quickly and, unlike the language model's, ship calibrated.
        pub fn prefix() -> String {
            std::env::var("ORANGU_NPU_BLOCK").unwrap_or_else(|_| "v.blk.0".into())
        }

        pub fn path() -> std::path::PathBuf {
            std::env::temp_dir().join("orangu-npu-real-ffn.npu")
        }

        /// Where the three-graph form's artifact goes, kept apart from the
        /// one-graph form's so a stale one cannot be read as the other.
        /// The gate activation this model uses — GELU for Gemma, SiLU for
        /// the llama family. Mirrors `npu_tool::host_activation`.
        pub fn activation(path: &std::path::Path) -> crate::npu_ort::HostActivation {
            use crate::npu_ort::HostActivation;
            let arch = crate::gguf::GgufFile::open(path)
                .ok()
                .and_then(|g| {
                    g.metadata
                        .iter()
                        .find(|(k, _)| k == "general.architecture")
                        .map(|(_, v)| v.display(1))
                })
                .unwrap_or_default();
            if arch.contains("gemma") {
                HostActivation::Gelu
            } else {
                HostActivation::Silu
            }
        }

        pub fn host_path() -> std::path::PathBuf {
            std::env::temp_dir().join("orangu-npu-real-ffn-host.npu")
        }

        pub struct Block {
            pub gate: Vec<f32>,
            pub up: Vec<f32>,
            pub down: Vec<f32>,
            pub d_model: usize,
            pub d_ff: usize,
            pub range: (f32, f32),
        }

        pub fn read(path: &std::path::Path) -> anyhow::Result<Block> {
            let gguf = crate::gguf::GgufFile::open(path)?;
            let prefix = prefix();
            let named = |suffix: &str| format!("{prefix}.{suffix}.weight");

            let dims = |name: &str| -> anyhow::Result<(usize, usize)> {
                let info = gguf
                    .tensors
                    .iter()
                    .find(|t| t.name == name)
                    .ok_or_else(|| anyhow::anyhow!("{name} is not in {}", path.display()))?;
                Ok((info.dims[0] as usize, info.dims[1] as usize))
            };
            let (d_model, d_ff) = dims(&named("ffn_gate"))?;

            // The activation range the model itself was calibrated with,
            // where it carries one.
            let range = match (
                gguf.read_scalar(path, &format!("{prefix}.ffn_gate.input_min")),
                gguf.read_scalar(path, &format!("{prefix}.ffn_gate.input_max")),
            ) {
                (Some(lo), Some(hi)) if hi > lo => (lo, hi),
                _ => (-1.0, 1.0),
            };

            let mut gate = gguf.read_tensor(path, &named("ffn_gate"))?;
            let mut up = gguf.read_tensor(path, &named("ffn_up"))?;
            let mut down = gguf.read_tensor(path, &named("ffn_down"))?;

            // `ORANGU_FFN_DFF` narrows the block to its first `n` hidden
            // units. Not a model anyone would run — a way to ask whether
            // this device's behaviour depends on `d_ff` while holding every
            // other property of a real block fixed.
            let mut d_ff = d_ff;
            if let Some(want) = std::env::var("ORANGU_FFN_DFF")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|w| *w > 0 && *w < d_ff)
            {
                gate.truncate(want * d_model);
                up.truncate(want * d_model);
                // `down` is `[d_model][d_ff]`, so each row is cut, not the tail.
                let mut narrowed = Vec::with_capacity(d_model * want);
                for o in 0..d_model {
                    narrowed.extend_from_slice(&down[o * d_ff..o * d_ff + want]);
                }
                down = narrowed;
                d_ff = want;
            }

            // `ORANGU_FFN_DMODEL` does the same to the feature dimension:
            // `gate`/`up` keep the first `n` of each row, `down` keeps its
            // first `n` rows.
            let mut d_model = d_model;
            if let Some(want) = std::env::var("ORANGU_FFN_DMODEL")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|w| *w > 0 && *w < d_model)
            {
                let narrow_rows = |v: &[f32], rows: usize| -> Vec<f32> {
                    let mut out = Vec::with_capacity(rows * want);
                    for r in 0..rows {
                        out.extend_from_slice(&v[r * d_model..r * d_model + want]);
                    }
                    out
                };
                gate = narrow_rows(&gate, d_ff);
                up = narrow_rows(&up, d_ff);
                down.truncate(want * d_ff);
                d_model = want;
            }

            Ok(Block {
                gate,
                up,
                down,
                d_model,
                d_ff,
                range,
            })
        }

        /// Calibration input spread across the range the model expects.
        pub fn calibration(d_model: usize, rows: usize, range: (f32, f32)) -> Vec<f32> {
            let mut seed = 0x51ed_1234u64;
            (0..rows * d_model)
                .map(|_| {
                    let unit = (super::pseudo_random(&mut seed) + 1.0) * 0.5;
                    range.0 + unit * (range.1 - range.0)
                })
                .collect()
        }
    }

    /// Compiling half: a real feed-forward block, fused into one graph.
    ///
    /// ```text
    /// export ORANGU_GGUF=/path/to/mmproj-gemma-4-E4B-it-Q8_0.gguf
    /// cargo test --release --lib npu_ort::tests::compiling_a_real_ffn -- --ignored --nocapture
    /// cargo test --release --lib npu_ort::tests::a_real_ffn -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "hardware; needs ORANGU_GGUF and its own process — see the module doc"]
    fn compiling_a_real_ffn() {
        let Some(path) = real_model::gguf() else {
            println!("set ORANGU_GGUF to a .gguf to run this");
            return;
        };
        let Some(ort) = NpuOrt::open() else {
            return;
        };
        let block = real_ffn::read(&path).expect("reading the block");
        let tokens = real_model::tokens();
        println!(
            "{}: {} -> {} -> {}, {tokens} tokens, activation range [{:.3}, {:.3}]",
            real_ffn::prefix(),
            block.d_model,
            block.d_ff,
            block.d_model,
            block.range.0,
            block.range.1
        );

        // As many rows as will be run: too few and the observed range is an
        // underestimate, which saturates at inference and costs more than
        // any amount of quantization coarseness.
        let calibration = real_ffn::calibration(block.d_model, tokens, block.range);
        let t0 = std::time::Instant::now();
        let compiled = match ort.compile_gated_ffn(
            &GatedFfn {
                d_model: block.d_model,
                d_ff: block.d_ff,
                gate: &block.gate,
                up: &block.up,
                down: &block.down,
            },
            tokens,
            // `ORANGU_FFN_ACT=none` drops the hand-spelled GELU, leaving the
            // two projections and the gating multiply. Not a network — a way
            // to ask whether the `Mul`/`Erf`/`Add`/`Mul` chain is where a
            // block's error comes from, by removing it and nothing else.
            match std::env::var("ORANGU_FFN_ACT").as_deref() {
                Ok("none") => GateActivation::None,
                Ok("sigmoid") => GateActivation::GeluSigmoid,
                Ok("scale") => GateActivation::ScaleOnly,
                Ok("fanout") => GateActivation::FanOut,
                Ok("sigmoid-only") => GateActivation::SigmoidOnly,
                Ok("split") => GateActivation::GeluSigmoidSplit,
                Ok("dup") => GateActivation::GeluSigmoidDup,
                Ok("two-input") => GateActivation::GeluSigmoidTwoInput,
                _ => GateActivation::Gelu,
            },
            &calibration,
        ) {
            Ok(c) => c,
            Err(e) => panic!("compiling a real FFN: {e}"),
        };
        println!(
            "compiled in {:.1} s, {:.1} MB",
            t0.elapsed().as_secs_f64(),
            compiled.binary().len() as f64 / 1e6
        );
        std::fs::write(real_ffn::path(), compiled.to_bytes()).expect("writing the artifact");
    }

    /// **The quantized block, simulated on the host.**
    ///
    /// Replays exactly what `build_gated_ffn_model` asks the device to do —
    /// the same quantize/dequantize at the same stages, from the same
    /// [`observed_quant`] ranges — in `f32`, and reports the error against
    /// the unquantized block. No device, no compile.
    ///
    /// It exists to make the accuracy question answerable in seconds. A
    /// compile is ~18 s and a device round trip needs a second process, so
    /// every "would this quantization help?" was costing minutes to ask.
    /// Here the same question is a loop. It earns that only if it agrees
    /// with the device, so it prints both the fused error and a per-stage
    /// breakdown, and the fused number is the one to check against
    /// `a_real_ffn_matches_a_cpu_reference`.
    ///
    /// `ORANGU_FFN_SLICES` splits `d_ff` into that many groups, each with
    /// its own quantization for the intermediates that are `d_ff` wide, and
    /// sums their partial `down` projections. That is per-channel
    /// quantization approximated with ops this provider does register — see
    /// [`NpuOrt::compile_gated_ffn`] for why the direct routes are closed.
    #[test]
    #[ignore = "needs ORANGU_GGUF; minutes of host arithmetic"]
    fn _scratch_simulated_gated_ffn_error() {
        let Some(path) = real_model::gguf() else {
            println!("set ORANGU_GGUF to a .gguf to run this");
            return;
        };
        let block = real_ffn::read(&path).expect("reading the block");
        let tokens = real_model::tokens();
        let slices: usize = std::env::var("ORANGU_FFN_SLICES")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|s| *s > 0)
            .unwrap_or(1);
        let (d_model, d_ff) = (block.d_model, block.d_ff);
        assert!(
            d_ff.is_multiple_of(slices),
            "d_ff {d_ff} does not divide into {slices} slices"
        );
        let group = d_ff / slices;
        let x = real_ffn::calibration(d_model, tokens, block.range);

        // `w` is `[out][in]` row-major, as the file stores it.
        let project = |input: &[f32], w: &[f32], in_dim: usize, out_dim: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; tokens * out_dim];
            for t in 0..tokens {
                for o in 0..out_dim {
                    let mut acc = 0.0f32;
                    for i in 0..in_dim {
                        acc += input[t * in_dim + i] * w[o * in_dim + i];
                    }
                    out[t * out_dim + o] = acc;
                }
            }
            out
        };
        // One round trip through a quantization, which is the only thing a
        // quantized tensor does to the values passing through it.
        let round = |v: &[f32], q: Quant| -> Vec<f32> {
            v.iter().map(|x| q.dequantize(q.quantize(*x))).collect()
        };

        // The model's own activation, not always GELU: a llama block is
        // SwiGLU, and simulating it as GELU would measure a network nobody
        // runs.
        let act = real_ffn::activation(&path);
        let want = {
            let g = project(&x, &block.gate, d_model, d_ff);
            let u = project(&x, &block.up, d_model, d_ff);
            let h: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(g, u)| act.apply_host(*g) * u)
                .collect();
            project(&h, &block.down, d_ff, d_model)
        };

        // Weights and input, quantized once, exactly as the graph holds them.
        let wq = |w: &[f32]| {
            let (lo, hi) = w
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            round(w, Quant::covering(lo, hi))
        };
        let (gate_w, up_w) = (wq(&block.gate), wq(&block.up));
        let half: Vec<f32> = block.down.iter().map(|v| v * 0.5).collect();
        let down_w = wq(&half);
        let xq = round(&x, observed_quant(&x));

        let gate_out = project(&xq, &gate_w, d_model, d_ff);
        let up_out = project(&xq, &up_w, d_model, d_ff);

        // What the `Erf` in the graph is actually handed. `erf` saturates
        // past about +/-2, so a range far wider than that spends most of a
        // `uint8`'s levels on two flat tails and leaves few for the part
        // that varies.
        let span = |v: &[f32]| {
            v.iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), x| (lo.min(*x), hi.max(*x)))
        };
        let (glo, ghi) = span(&gate_out);
        println!(
            "  gate_out [{glo:.3}, {ghi:.3}]  -> erf input [{:.3}, {:.3}]",
            glo * std::f32::consts::FRAC_1_SQRT_2,
            ghi * std::f32::consts::FRAC_1_SQRT_2,
        );

        // **The three-graph pipeline, as `compile_gated_ffn_host` builds
        // it**: two projections on the device, the activation exactly in
        // `f32` on the host, then `down` on the device. Every tensor that
        // crosses the boundary is rounded through its own `uint8`
        // quantization and nothing else is.
        //
        // This used to model the one-graph form, with the activation spelled
        // out in device ops and quantized at every step. That is not what
        // runs any more, so it was measuring a pipeline nobody uses.
        let mut got = vec![0.0f32; tokens * d_model];
        for s in 0..slices {
            let cols = s * group..(s + 1) * group;
            let take = |v: &[f32]| -> Vec<f32> {
                (0..tokens)
                    .flat_map(|t| v[t * d_ff + cols.start..t * d_ff + cols.end].to_vec())
                    .collect()
            };
            let g = round(&take(&gate_out), observed_quant(&take(&gate_out)));
            let u = round(&take(&up_out), observed_quant(&take(&up_out)));
            // Exact: this is the host's own arithmetic, not the device's.
            let hidden: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(g, u)| act.apply_host(*g) * u)
                .collect();
            let hidden = round(&hidden, observed_quant(&hidden));

            let mut dw = vec![0.0f32; d_model * group];
            for o in 0..d_model {
                dw[o * group..(o + 1) * group]
                    .copy_from_slice(&down_w[o * d_ff + cols.start..o * d_ff + cols.end]);
            }
            let partial = project(&hidden, &dw, group, d_model);
            for (acc, v) in got.iter_mut().zip(&partial) {
                *acc += v;
            }
        }
        let got = round(&got, observed_quant(&got));

        let magnitude = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = want
            .iter()
            .zip(&got)
            .fold(0.0f32, |m, (w, g)| m.max((w - g).abs()));
        println!(
            "simulated {} {tokens} tok x {d_model} x {d_ff}, {slices} slice(s): \
             worst {worst:.4} of {magnitude:.4} = {:.1}%",
            real_ffn::prefix(),
            100.0 * worst / magnitude,
        );
    }

    /// The vector `silu(gate) * up` against the scalar one it replaces.
    ///
    /// Same contract as the GELU twin: a length that is not a multiple of
    /// four so the tail is covered, and a range wide enough to reach where
    /// `sigmoid` saturates at both ends.
    #[test]
    fn the_vector_silu_matches_the_scalar_one() {
        let n = 1023;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 511.0) / 23.0).collect();
        let up: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.3).collect();

        let mut want = vec![0.0f32; n];
        super::silu_mul_scalar(&gate, &up, &mut want);
        let mut got = vec![0.0f32; n];
        super::silu_mul_into(&gate, &up, &mut got);

        for (i, (w, g)) in want.iter().zip(&got).enumerate() {
            assert!(
                (w - g).abs() <= 1e-5 * w.abs().max(1.0),
                "lane {i}: scalar {w}, vector {g} (gate {})",
                gate[i]
            );
        }
    }

    /// The vector `gelu(gate) * up` against the scalar one it replaces.
    ///
    /// Both spell Abramowitz and Stegun 7.1.26, so they agree to their
    /// shared approximation of `erf` rather than to two different ones; what
    /// this bounds is the vector `exp` and the lane arithmetic. Covers a
    /// length that is not a multiple of four, since the tail takes the
    /// scalar path.
    #[test]
    fn the_vector_gelu_matches_the_scalar_one() {
        let n = 1023;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 511.0) / 37.0).collect();
        let up: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();

        let mut want = vec![0.0f32; n];
        super::gelu_mul_scalar(&gate, &up, &mut want);
        let mut got = vec![0.0f32; n];
        super::gelu_mul_into(&gate, &up, &mut got);

        for (i, (w, g)) in want.iter().zip(&got).enumerate() {
            assert!(
                (w - g).abs() <= 1e-5 * w.abs().max(1.0),
                "lane {i}: scalar {w}, vector {g} (gate {}, up {})",
                gate[i],
                up[i]
            );
        }
    }

    /// The vector `exp` over the range `erf` actually hands it — `-x*x` for
    /// the `x` a gate produces, so zero down to well below where the result
    /// stops mattering.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn the_vector_exp_matches_libm() {
        use std::arch::aarch64::*;
        for start in [-80.0f32, -20.0, -5.0, -0.7, 0.0] {
            let lanes = [start, start + 0.1, start + 0.25, start + 0.5];
            // SAFETY: NEON is baseline on aarch64 and `lanes` is four wide.
            let got: [f32; 4] = unsafe {
                let v = super::exp_neon(vld1q_f32(lanes.as_ptr()));
                let mut out = [0.0f32; 4];
                vst1q_f32(out.as_mut_ptr(), v);
                out
            };
            for (x, g) in lanes.iter().zip(&got) {
                let want = x.exp();
                // **Relative**, and 1e-5 of it. `exp` spans 38 decades over
                // the range that matters, so an absolute bound is either
                // meaningless at the top or unmeetable at the bottom. The
                // measured worst case is about 3e-6 relative, which for a
                // degree-5 polynomial is a few ulp of `f32` and orders of
                // magnitude finer than the `uint8` quantization that follows.
                assert!(
                    (want - g).abs() <= 1e-5 * want.abs(),
                    "exp({x}): want {want}, got {g}"
                );
            }
        }
    }

    /// **A real feed-forward block as three graphs, with the activation on
    /// the host.** Compiles it; `a_real_host_ffn_matches_a_cpu_reference`
    /// runs it in a second process.
    #[test]
    #[ignore = "hardware; writes an artifact for the run test"]
    fn compiling_a_real_host_ffn() {
        let Some(path) = real_model::gguf() else {
            println!("set ORANGU_GGUF to a .gguf to run this");
            return;
        };
        let Some(ort) = NpuOrt::open() else {
            return;
        };
        let block = real_ffn::read(&path).expect("reading the block");
        let tokens = real_model::tokens();
        println!(
            "{}: {} -> {} -> {}, {tokens} tokens",
            real_ffn::prefix(),
            block.d_model,
            block.d_ff,
            block.d_model,
        );
        let calibration = real_ffn::calibration(block.d_model, tokens, block.range);
        let started = std::time::Instant::now();
        let compiled = ort
            .compile_gated_ffn_host(
                &GatedFfn {
                    d_model: block.d_model,
                    d_ff: block.d_ff,
                    gate: &block.gate,
                    up: &block.up,
                    down: &block.down,
                },
                tokens,
                &calibration,
                // From the file, the same way `npu_tool` picks it, so this
                // test compiles a llama block as SwiGLU rather than
                // silently as Gemma's GELU.
                real_ffn::activation(&path),
            )
            .expect("compiling a host-activation FFN");
        println!(
            "compiled in {:.1} s, {}",
            started.elapsed().as_secs_f64(),
            crate::format::format_bytes(compiled.binary_len() as u64),
        );
        std::fs::write(real_ffn::host_path(), compiled.to_bytes()).expect("writing the artifact");
    }

    /// **The three-graph block against the same block in `f32`.**
    ///
    /// The one-graph form fails this at about 50% (see
    /// [`NpuOrt::compile_gated_ffn`]). Same comparison, so the numbers stand
    /// directly against each other.
    #[test]
    #[ignore = "hardware; needs the artifact from compiling_a_real_host_ffn"]
    fn a_real_host_ffn_matches_a_cpu_reference() {
        let Some(path) = real_model::gguf() else {
            return;
        };
        let Ok(bytes) = std::fs::read(real_ffn::host_path()) else {
            println!("no artifact; run the compiling test first");
            return;
        };
        let compiled = CompiledGatedFfn::from_bytes(&bytes).expect("reading the artifact");
        let Some(runtime) = crate::npu::NpuRuntime::open() else {
            return;
        };
        let block = real_ffn::read(&path).expect("reading the block");
        let tokens = compiled.max_tokens();
        let ffn = compiled.bind(&runtime).expect("binding");

        let x = real_ffn::calibration(block.d_model, tokens, block.range);
        let mut got = Vec::new();
        let mut samples = Vec::new();
        for _ in 0..20 {
            let t = std::time::Instant::now();
            ffn.forward_into(&x, tokens, &mut got).expect("forward");
            samples.push(t.elapsed().as_secs_f64());
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let per = samples[samples.len() / 2];

        let project = |input: &[f32], w: &[f32], in_dim: usize, out_dim: usize| {
            let mut out = vec![0.0f32; tokens * out_dim];
            for t in 0..tokens {
                for o in 0..out_dim {
                    let mut acc = 0.0f32;
                    for i in 0..in_dim {
                        acc += input[t * in_dim + i] * w[o * in_dim + i];
                    }
                    out[t * out_dim + o] = acc;
                }
            }
            out
        };
        let gate_out = project(&x, &block.gate, block.d_model, block.d_ff);
        let up_out = project(&x, &block.up, block.d_model, block.d_ff);
        let hidden: Vec<f32> = gate_out
            .iter()
            .zip(&up_out)
            .map(|(g, u)| gelu(*g) * u)
            .collect();
        let want = project(&hidden, &block.down, block.d_ff, block.d_model);

        let magnitude = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = want
            .iter()
            .zip(&got)
            .fold(0.0f32, |m, (w, g)| m.max((w - g).abs()));
        let flops = 3.0 * 2.0 * tokens as f64 * block.d_model as f64 * block.d_ff as f64;
        println!(
            "host-activation FFN {tokens} tok x {} x {}: {:.3} ms ({:.1} GFLOP/s), worst {worst:.4} of {magnitude:.4} = {:.1}%",
            block.d_model,
            block.d_ff,
            per * 1000.0,
            flops / per / 1e9,
            100.0 * worst / magnitude,
        );
        assert!(
            worst <= 0.2 * magnitude,
            "three-graph FFN differs by {worst}, more than 20% of {magnitude}"
        );
    }

    /// **A real feed-forward block on the NPU, against the same block in
    /// `f32`.**
    #[test]
    #[ignore = "hardware; needs the artifact from compiling_a_real_ffn"]
    fn a_real_ffn_matches_a_cpu_reference() {
        let Some(path) = real_model::gguf() else {
            return;
        };
        let Ok(bytes) = std::fs::read(real_ffn::path()) else {
            println!("no artifact; run the compiling test first");
            return;
        };
        let compiled = CompiledLinear::from_bytes(&bytes).expect("reading the artifact");
        let Some(runtime) = crate::npu::NpuRuntime::open() else {
            return;
        };
        let block = real_ffn::read(&path).expect("reading the block");
        let tokens = compiled.max_tokens();
        let fused = compiled.into_bound(&runtime).expect("binding");

        let x = real_ffn::calibration(block.d_model, tokens, block.range);
        let mut samples = Vec::new();
        let mut got = Vec::new();
        for _ in 0..20 {
            let t = std::time::Instant::now();
            fused.forward_into(&x, tokens, &mut got).expect("forward");
            samples.push(t.elapsed().as_secs_f64());
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let per = samples[samples.len() / 2];

        let project = |input: &[f32], w: &[f32], in_dim: usize, out_dim: usize| {
            let mut out = vec![0.0f32; tokens * out_dim];
            for t in 0..tokens {
                for o in 0..out_dim {
                    let mut acc = 0.0f32;
                    for i in 0..in_dim {
                        acc += input[t * in_dim + i] * w[o * in_dim + i];
                    }
                    out[t * out_dim + o] = acc;
                }
            }
            out
        };
        let gate_out = project(&x, &block.gate, block.d_model, block.d_ff);
        let up_out = project(&x, &block.up, block.d_model, block.d_ff);
        // Matches whatever `compiling_a_real_ffn` was told to build.
        // The reference is true GELU either way. `GeluSigmoid` is an
        // approximation of it worth about 0.02 absolute, so checking it
        // against itself would hide that; the point is whether the whole
        // block lands close to the function it is meant to compute.
        let act = std::env::var("ORANGU_FFN_ACT").unwrap_or_default();
        let hidden: Vec<f32> = gate_out
            .iter()
            .zip(&up_out)
            .map(|(g, u)| match act.as_str() {
                "none" => *g * u,
                "scale" => *g * GELU_SIGMOID_K * u,
                "fanout" => *g * (*g * GELU_SIGMOID_K) * u,
                "sigmoid-only" => u / (1.0 + (-*g).exp()),
                _ => gelu(*g) * u,
            })
            .collect();
        let want = project(&hidden, &block.down, block.d_ff, block.d_model);

        // Both failing spellings miss by about half the signal, which is what
        // a factor of two looks like rather than a miscomputed curve. The
        // `Erf` spelling folds GELU's `* 0.5` into the `down` weights; if the
        // provider pattern-matches the subgraph and applies its own halving
        // too, the result is half of what it should be. Report the error
        // against `want` rescaled, so a clean match at one of these says so.
        for scale in [0.5f32, 2.0] {
            let worst_scaled = want
                .iter()
                .zip(&got)
                .fold(0.0f32, |m, (w, g)| m.max((w * scale - g).abs()));
            println!(
                "  against {scale}x the reference: worst {worst_scaled:.4} of {:.4}",
                want.iter().fold(0.0f32, |m, v| m.max((v * scale).abs()))
            );
        }

        // The sigmoid spellings compute `x * sigmoid(1.702x)`, which is an
        // *approximation* of GELU worth about 0.02 per activation. Against
        // true GELU they must therefore differ, so the question is whether
        // they differ by more than that accounts for — report both.
        if matches!(act.as_str(), "sigmoid" | "split" | "dup" | "two-input") {
            let approx: Vec<f32> = gate_out
                .iter()
                .zip(&up_out)
                .map(|(g, u)| (*g / (1.0 + (-GELU_SIGMOID_K * *g).exp())) * u)
                .collect();
            let want_approx = project(&approx, &block.down, block.d_ff, block.d_model);
            let worst_approx = want_approx
                .iter()
                .zip(&got)
                .fold(0.0f32, |m, (w, g)| m.max((w - g).abs()));
            let mag_approx = want_approx.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            println!(
                "  against `x*sigmoid(1.702x)` itself: worst {worst_approx:.4} of {mag_approx:.4} = {:.1}%",
                100.0 * worst_approx / mag_approx
            );
        }

        let magnitude = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Three projections, so three times the arithmetic of one.
        let flops = 3.0 * 2.0 * tokens as f64 * block.d_model as f64 * block.d_ff as f64;
        println!(
            "real FFN {tokens} tok x {} x {}: {:.3} ms ({:.1} GFLOP/s)",
            block.d_model,
            block.d_ff,
            per * 1000.0,
            flops / per / 1e9
        );
        println!(
            "  outputs reach {magnitude:.3}, worst error {worst:.4} ({:.2}% of that)",
            100.0 * worst / magnitude.max(f32::MIN_POSITIVE)
        );

        assert!(got.iter().any(|v| *v != 0.0), "every output was zero");
        // 20%, and this threshold is a measurement rather than a target: a
        // real block through eight quantized stages lands near 11%, and no
        // amount of calibration moved it far — 4 rows and 196 rows agree,
        // and narrowing the margin made it worse. It is set where it is to
        // catch a *structural* mistake, which misses by 90% and more, not to
        // certify accuracy. What this path costs in precision is the printed
        // number above, and the driving input here is the worst case: values
        // spread uniformly across the whole calibrated range, where real
        // activations concentrate.
        assert!(
            worst <= 0.20 * magnitude.max(f32::MIN_POSITIVE),
            "real FFN differs by {worst}, more than 20% of {magnitude} — \
             that is structural, not quantization"
        );
    }

    /// Compiling half: take a real Q8_0 projection to an NPU artifact.
    ///
    /// ```text
    /// export ORANGU_GGUF=/path/to/mmproj-gemma-4-E4B-it-Q8_0.gguf
    /// cargo test --release --lib npu_ort::tests::compiling_a_real -- --ignored --nocapture
    /// cargo test --release --lib npu_ort::tests::a_real_projection -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "hardware; needs ORANGU_GGUF and its own process — see the module doc"]
    fn compiling_a_real_projection_from_a_gguf() {
        let Some(path) = real_model::gguf() else {
            println!("set ORANGU_GGUF to a .gguf to run this");
            return;
        };
        let Some(ort) = NpuOrt::open() else {
            return;
        };
        let layer = real_model::layer(&path).expect("reading the layer");
        let (k, n, range) = (layer.in_features, layer.out_features, layer.range);
        let tokens = real_model::tokens();
        println!(
            "{}: {k} x {n}, activation range [{:.4}, {:.4}]",
            real_model::tensor(),
            range.0,
            range.1
        );

        let t0 = std::time::Instant::now();
        let compiled = match ort.compile_linear(&layer.weights, k, n, tokens, range) {
            Ok(c) => c,
            Err(e) => panic!("compiling {}: {e}", real_model::tensor()),
        };
        println!(
            "compiled {} tokens in {:.0} ms, {} bytes",
            tokens,
            t0.elapsed().as_secs_f64() * 1000.0,
            compiled.binary().len()
        );
        std::fs::write(real_model::artifact(), compiled.to_bytes()).expect("writing the artifact");
    }

    /// **Real weights, real calibration, real shape — checked against `f32`.**
    ///
    /// The tolerance is the device's own quantization step rather than a
    /// number chosen to make this pass, so a regression in the mapping shows
    /// up as a failure instead of a slightly worse answer.
    #[test]
    #[ignore = "hardware; needs the artifact from compiling_a_real_projection_from_a_gguf"]
    fn a_real_projection_matches_a_cpu_reference() {
        let Some(path) = real_model::gguf() else {
            return;
        };
        let Ok(bytes) = std::fs::read(real_model::artifact()) else {
            println!("no artifact; run the compiling test first");
            return;
        };
        let compiled = CompiledLinear::from_bytes(&bytes).expect("reading the artifact");
        let Some(runtime) = crate::npu::NpuRuntime::open() else {
            return;
        };
        let layer_info = real_model::layer(&path).expect("reading the layer");
        let (weights, k, n, range) = (
            layer_info.weights,
            layer_info.in_features,
            layer_info.out_features,
            layer_info.range,
        );
        let tokens = compiled.max_tokens();
        let layer = compiled.into_bound(&runtime).expect("binding");

        // Activations spread across the range the model says to expect.
        let mut seed = 0x9e3779b9u64;
        let x: Vec<f32> = (0..tokens * k)
            .map(|_| {
                let unit = (pseudo_random(&mut seed) + 1.0) * 0.5;
                range.0 + unit * (range.1 - range.0)
            })
            .collect();

        let t0 = std::time::Instant::now();
        let got = layer.forward(&x, tokens).expect("forward");
        let elapsed = t0.elapsed().as_secs_f64();

        let mut want = vec![0.0f32; tokens * n];
        for t in 0..tokens {
            for o in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += x[t * k + i] * weights[o * k + i];
                }
                want[t * n + o] = acc;
            }
        }

        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let tolerance = 2.0 * layer.output_scale() + (k as f32) * layer.input_scale() * 0.5;
        // An absolute error means nothing without the scale it sits on, so
        // report it against the largest value the layer actually produced.
        let magnitude = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let flops = 2.0 * tokens as f64 * k as f64 * n as f64;
        println!(
            "{tokens} tokens x {k} x {n}: {:.3} ms ({:.1} GFLOP/s)",
            elapsed * 1000.0,
            flops / elapsed / 1e9
        );
        println!(
            "  outputs reach {magnitude:.2}; worst error {worst:.4} ({:.2}% of that), tolerance {tolerance:.4}",
            100.0 * worst / magnitude.max(f32::MIN_POSITIVE)
        );

        assert!(got.iter().any(|v| *v != 0.0), "every output was zero");
        assert!(
            worst <= tolerance,
            "real projection differs by {worst}, tolerance {tolerance}"
        );
    }

    /// A small gated feed-forward block, the shape both Gemma 4 models use.
    mod ffn_fixture {
        pub const D_MODEL: usize = 64;
        pub const D_FF: usize = 128;
        pub const TOKENS: usize = 8;

        /// Pseudo-random rather than periodic, and deliberately so. An
        /// earlier version of this fixture used short repeating patterns,
        /// which correlate across a projection and cancel: the outputs came
        /// out near zero and the relative error looked enormous when what was
        /// really small was the signal. Weights that do not line up with each
        /// other are both more realistic and a fairer measurement.
        fn weights(count: usize, seed: u64, scale: f32) -> Vec<f32> {
            let mut state = seed;
            (0..count)
                .map(|_| super::pseudo_random(&mut state) * scale)
                .collect()
        }

        pub fn gate() -> Vec<f32> {
            weights(D_FF * D_MODEL, 0x1234_5678, 0.12)
        }

        pub fn up() -> Vec<f32> {
            weights(D_FF * D_MODEL, 0x8765_4321, 0.15)
        }

        pub fn down() -> Vec<f32> {
            weights(D_MODEL * D_FF, 0xfeed_face, 0.10)
        }

        pub fn input() -> Vec<f32> {
            weights(TOKENS * D_MODEL, 0x0bad_c0de, 0.8)
        }

        /// `ORANGU_NPU_ACT=none` drops the activation, which is the first
        /// thing to try when the block will not compile.
        pub fn activation() -> super::GateActivation {
            match std::env::var("ORANGU_NPU_ACT").as_deref() {
                Ok("none") => super::GateActivation::None,
                _ => super::GateActivation::Gelu,
            }
        }

        pub fn path() -> std::path::PathBuf {
            std::env::temp_dir().join("orangu-npu-ffn.npu")
        }
    }

    /// Compiling half of the gated-FFN round trip.
    #[test]
    #[ignore = "hardware; must run in its own process — see the module doc"]
    fn compiling_a_gated_ffn() {
        use ffn_fixture::{D_FF, D_MODEL, TOKENS};
        let Some(ort) = NpuOrt::open() else {
            return;
        };
        let (gate, up, down) = (ffn_fixture::gate(), ffn_fixture::up(), ffn_fixture::down());
        let activation = ffn_fixture::activation();

        let t0 = std::time::Instant::now();
        let compiled = match ort.compile_gated_ffn(
            &GatedFfn {
                d_model: D_MODEL,
                d_ff: D_FF,
                gate: &gate,
                up: &up,
                down: &down,
            },
            TOKENS,
            activation,
            &ffn_fixture::input(),
        ) {
            Ok(c) => c,
            Err(e) => panic!("compiling a gated FFN with {activation:?}: {e}"),
        };
        println!(
            "gated FFN ({activation:?}) {D_MODEL}->{D_FF}->{D_MODEL}: compiled in {:.0} ms, {} bytes",
            t0.elapsed().as_secs_f64() * 1000.0,
            compiled.binary().len()
        );
        std::fs::write(ffn_fixture::path(), compiled.to_bytes()).expect("writing the artifact");
    }

    /// **A whole feed-forward block, against the same block in `f32`.**
    ///
    /// The diamond — two projections off one input, joined by a multiply —
    /// is the part a chain of layers cannot express, and the activation is
    /// spelled out from `Erf` because the provider registers no `Gelu`. Both
    /// are checked here rather than assumed to have compiled into what they
    /// were meant to be.
    #[test]
    #[ignore = "hardware; needs the artifact from compiling_a_gated_ffn"]
    fn a_gated_ffn_matches_a_cpu_reference() {
        use ffn_fixture::{D_FF, D_MODEL, TOKENS};
        let Ok(bytes) = std::fs::read(ffn_fixture::path()) else {
            println!("no FFN artifact; run the compiling test first");
            return;
        };
        let compiled = CompiledLinear::from_bytes(&bytes).expect("reading the artifact");
        let Some(runtime) = crate::npu::NpuRuntime::open() else {
            return;
        };
        let block = compiled.into_bound(&runtime).expect("binding the FFN");

        let (gate, up, down) = (ffn_fixture::gate(), ffn_fixture::up(), ffn_fixture::down());
        let x = ffn_fixture::input();
        let activation = ffn_fixture::activation();
        let got = block.forward(&x, TOKENS).expect("forward failed");
        assert_eq!(got.len(), TOKENS * D_MODEL);

        let project = |input: &[f32], w: &[f32], in_dim: usize, out_dim: usize| {
            let mut out = vec![0.0f32; TOKENS * out_dim];
            for t in 0..TOKENS {
                for o in 0..out_dim {
                    let mut acc = 0.0f32;
                    for i in 0..in_dim {
                        acc += input[t * in_dim + i] * w[o * in_dim + i];
                    }
                    out[t * out_dim + o] = acc;
                }
            }
            out
        };
        let gate_out = project(&x, &gate, D_MODEL, D_FF);
        let up_out = project(&x, &up, D_MODEL, D_FF);
        let hidden: Vec<f32> = gate_out
            .iter()
            .zip(&up_out)
            .map(|(g, u)| {
                let activated = match activation {
                    GateActivation::None => *g,
                    GateActivation::Gelu => gelu(*g),
                    GateActivation::GeluSigmoid => *g / (1.0 + (-GELU_SIGMOID_K * *g).exp()),
                    GateActivation::ScaleOnly => *g * GELU_SIGMOID_K,
                    GateActivation::FanOut => *g * (*g * GELU_SIGMOID_K),
                    GateActivation::SigmoidOnly => 1.0 / (1.0 + (-*g).exp()),
                    GateActivation::GeluSigmoidSplit
                    | GateActivation::GeluSigmoidDup
                    | GateActivation::GeluSigmoidTwoInput => {
                        *g / (1.0 + (-GELU_SIGMOID_K * *g).exp())
                    }
                };
                activated * u
            })
            .collect();
        let want = project(&hidden, &down, D_FF, D_MODEL);

        let magnitude = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "gated FFN ({activation:?}): outputs reach {magnitude:.3}, worst error {worst:.4} ({:.1}% of that)",
            100.0 * worst / magnitude.max(f32::MIN_POSITIVE)
        );

        assert!(got.iter().any(|v| *v != 0.0), "every output was zero");
        // 2%, against a measured 0.5%. Loose enough that ordinary
        // quantization noise never trips it, tight enough that the two
        // structural mistakes this test has already caught — a mis-wired
        // diamond, and a GELU whose constants were folded into
        // dequantization scales — both fail it by a wide margin.
        assert!(
            worst <= 0.02 * magnitude.max(f32::MIN_POSITIVE),
            "gated FFN differs by {worst}, more than 2% of {magnitude}"
        );
    }

    /// The fused-versus-separate comparison's shared shape and weights.
    mod fusion_bench {
        pub const WIDTH: usize = 512;
        pub const TOKENS: usize = 128;
        pub const LAYERS: usize = 3;

        pub fn layer(i: usize) -> Vec<f32> {
            (0..WIDTH * WIDTH)
                .map(|j| (((j + i * 7) % 11) as f32 - 5.0) * 0.01)
                .collect()
        }

        pub fn input() -> Vec<f32> {
            (0..TOKENS * WIDTH)
                .map(|i| ((i % 9) as f32 - 4.0) * 0.2)
                .collect()
        }

        pub fn fused_path() -> std::path::PathBuf {
            std::env::temp_dir().join("orangu-npu-fused.npu")
        }

        pub fn separate_path(i: usize) -> std::path::PathBuf {
            std::env::temp_dir().join(format!("orangu-npu-separate-{i}.npu"))
        }
    }

    /// Compiles both halves of the fused-versus-separate comparison.
    ///
    /// ```text
    /// cargo test --release --lib npu_ort::tests::compiling_the_fusion -- --ignored --nocapture
    /// cargo test --release --lib npu_ort::tests::fused_versus_separate -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn compiling_the_fusion_benchmark() {
        use fusion_bench::{LAYERS, TOKENS, WIDTH};
        let Some(ort) = NpuOrt::open() else {
            println!("no NPU compiler on this machine");
            return;
        };
        let layers: Vec<Vec<f32>> = (0..LAYERS).map(fusion_bench::layer).collect();
        let borrowed: Vec<&[f32]> = layers.iter().map(|l| l.as_slice()).collect();
        let calibration = fusion_bench::input();

        let dims = vec![WIDTH; LAYERS + 1];
        let t0 = std::time::Instant::now();
        match ort.compile_stack(&dims, &borrowed, TOKENS, &calibration) {
            Ok(c) => {
                std::fs::write(fusion_bench::fused_path(), c.to_bytes()).expect("write");
                println!(
                    "fused {LAYERS} layers: compiled in {:>6.0} ms, {} bytes",
                    t0.elapsed().as_secs_f64() * 1000.0,
                    c.binary().len()
                );
            }
            Err(e) => println!("fused compile FAILED: {e}"),
        }

        for (i, layer) in layers.iter().enumerate() {
            let t = std::time::Instant::now();
            match ort.compile_stack(&[WIDTH, WIDTH], &[layer], TOKENS, &calibration) {
                Ok(c) => {
                    std::fs::write(fusion_bench::separate_path(i), c.to_bytes()).expect("write");
                    println!(
                        "separate layer {i}: compiled in {:>6.0} ms, {} bytes",
                        t.elapsed().as_secs_f64() * 1000.0,
                        c.binary().len()
                    );
                }
                Err(e) => println!("separate layer {i} compile FAILED: {e}"),
            }
        }
    }

    /// **Does folding consecutive layers into one graph pay?**
    ///
    /// A single projection spends more time converting at the host boundary
    /// than computing on the device, so the answer should be yes — a fused
    /// chain crosses that boundary once instead of once per layer. This
    /// measures it rather than assuming it.
    #[test]
    #[ignore = "benchmark; needs the artifacts from compiling_the_fusion_benchmark"]
    fn fused_versus_separate_throughput() {
        use fusion_bench::{LAYERS, TOKENS, WIDTH};
        let Some(runtime) = crate::npu::NpuRuntime::open() else {
            println!("no NPU runtime on this machine");
            return;
        };
        let x = fusion_bench::input();

        let time = |label: &str, run: &mut dyn FnMut()| {
            let mut samples = Vec::new();
            for _ in 0..30 {
                let t = std::time::Instant::now();
                run();
                samples.push(t.elapsed().as_secs_f64());
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let per = samples[samples.len() / 2];
            let flops = 2.0 * TOKENS as f64 * WIDTH as f64 * WIDTH as f64 * LAYERS as f64;
            println!(
                "{label:>22}: {:>7.3} ms  ({:>7.2} GFLOP/s)",
                per * 1000.0,
                flops / per / 1e9
            );
            per
        };

        let fused = std::fs::read(fusion_bench::fused_path())
            .ok()
            .and_then(|b| CompiledLinear::from_bytes(&b).ok())
            .and_then(|c| c.into_bound(&runtime).ok());
        let separate: Vec<NpuLinear> = (0..LAYERS)
            .filter_map(|i| {
                std::fs::read(fusion_bench::separate_path(i))
                    .ok()
                    .and_then(|b| CompiledLinear::from_bytes(&b).ok())
                    .and_then(|c| c.into_bound(&runtime).ok())
            })
            .collect();

        let mut fused_time = None;
        if let Some(fused) = &fused {
            let mut out = Vec::new();
            fused_time = Some(time("fused, one graph", &mut || {
                fused
                    .forward_into(&x, TOKENS, &mut out)
                    .expect("fused forward");
            }));
        } else {
            println!("no fused artifact; run the compiling test first");
        }

        if separate.len() == LAYERS {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            let separate_time = time("separate, one per layer", &mut || {
                separate[0]
                    .forward_into(&x, TOKENS, &mut a)
                    .expect("layer 0");
                for layer in &separate[1..] {
                    layer.forward_into(&a, TOKENS, &mut b).expect("layer");
                    std::mem::swap(&mut a, &mut b);
                }
            });
            if let Some(fused_time) = fused_time {
                println!(
                    "{:>22}: {:.2}x",
                    "fusion speedup",
                    separate_time / fused_time
                );
            }
        } else {
            println!(
                "only {} of {LAYERS} separate artifacts; run the compiling test first",
                separate.len()
            );
        }
    }

    /// Compiles the benchmark shapes and writes them beside each other, for
    /// [`throughput_at_transformer_shapes`] to pick up in another process.
    ///
    /// ```text
    /// cargo test --lib npu_ort::tests::compiling_the_benchmark -- --ignored --nocapture
    /// cargo test --lib npu_ort::tests::throughput -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn compiling_the_benchmark_shapes() {
        let Some(ort) = NpuOrt::open() else {
            println!("no NPU compiler on this machine");
            return;
        };
        for (tokens, k, n) in bench_shapes() {
            let weights = bench_weights(k, n);
            let t0 = std::time::Instant::now();
            match ort.compile_linear(&weights, k, n, tokens, (-1.0, 1.0)) {
                Ok(compiled) => {
                    let path = bench_artifact(tokens, k, n);
                    std::fs::write(&path, compiled.to_bytes()).expect("writing the artifact");
                    println!(
                        "{tokens:>4} tok x {k:>4} x {n:<4}  compiled in {:>6.0} ms -> {}",
                        t0.elapsed().as_secs_f64() * 1000.0,
                        path.display()
                    );
                }
                Err(e) => println!("{tokens:>4}x{k}x{n}  compile FAILED: {e}"),
            }
        }
    }

    /// Throughput at transformer-shaped projections, against a plain f32 CPU
    /// matmul for scale.
    ///
    /// Not a correctness test — it asserts almost nothing, because the useful
    /// output is the numbers. Note what the CPU column is and is not: a naive
    /// scalar triple loop, *not* orangu's own CPU backend. It is there to give
    /// the NPU figure a familiar unit, not to claim a speedup over anything
    /// anyone would ship.
    #[test]
    #[ignore = "benchmark; needs the artifacts from compiling_the_benchmark_shapes"]
    fn throughput_at_transformer_shapes() {
        let Some(runtime) = crate::npu::NpuRuntime::open() else {
            println!("no NPU runtime on this machine");
            return;
        };
        for (tokens, k, n) in bench_shapes() {
            let path = bench_artifact(tokens, k, n);
            let Ok(bytes) = std::fs::read(&path) else {
                println!("{tokens:>4}x{k}x{n}  no artifact; run the compiling test first");
                continue;
            };
            let compiled = match CompiledLinear::from_bytes(&bytes) {
                Ok(c) => c,
                Err(e) => {
                    println!("{tokens:>4}x{k}x{n}  unreadable artifact: {e}");
                    continue;
                }
            };
            let load = std::time::Instant::now();
            let linear = match compiled.bind(&runtime) {
                Ok(l) => l,
                Err(e) => {
                    println!("{tokens:>4}x{k}x{n}  bind FAILED: {e}");
                    continue;
                }
            };
            let load = load.elapsed();

            let weights = bench_weights(k, n);
            let x: Vec<f32> = (0..tokens * k)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
                .collect();

            // Steady state, not a single sample: the first inference pays
            // one-off costs the rest do not.
            let mut samples = Vec::new();
            for _ in 0..50 {
                let t = std::time::Instant::now();
                if let Err(e) = linear.forward(&x, tokens) {
                    println!("{tokens:>4}x{k}x{n}  forward FAILED: {e}");
                    break;
                }
                samples.push(t.elapsed().as_secs_f64());
            }
            if samples.is_empty() {
                continue;
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let per = samples[samples.len() / 2];

            // A naive f32 CPU matmul over the same shape, for scale.
            let t2 = std::time::Instant::now();
            let mut out = vec![0.0f32; tokens * n];
            for t in 0..tokens {
                for o in 0..n {
                    let mut acc = 0.0f32;
                    for i in 0..k {
                        acc += x[t * k + i] * weights[o * k + i];
                    }
                    out[t * n + o] = acc;
                }
            }
            let cpu = t2.elapsed().as_secs_f64();

            let flops = 2.0 * tokens as f64 * k as f64 * n as f64;
            println!(
                "{tokens:>4} tok x {k:>4} x {n:<4}  load {:>6.2} ms  npu {:>7.3} ms ({:>7.2} GFLOP/s)  naive cpu {:>9.3} ms",
                load.as_secs_f64() * 1000.0,
                per * 1000.0,
                flops / per / 1e9,
                cpu * 1000.0,
            );
        }
    }

    /// Shapes come from `BENCH_SHAPES` as `tokens,k,n;tokens,k,n`, so a
    /// single shape can be run in isolation.
    fn bench_shapes() -> Vec<(usize, usize, usize)> {
        std::env::var("BENCH_SHAPES")
            .unwrap_or_else(|_| "8,64,32;32,512,512;128,512,512;256,1024,1024".into())
            .split(';')
            .filter_map(|t| {
                let v: Vec<usize> = t.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                (v.len() == 3).then(|| (v[0], v[1], v[2]))
            })
            .collect()
    }

    fn bench_weights(k: usize, n: usize) -> Vec<f32> {
        (0..n * k).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect()
    }

    fn bench_artifact(tokens: usize, k: usize, n: usize) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("orangu-npu-bench-{tokens}x{k}x{n}.npu"))
    }
}
