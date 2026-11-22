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

//! Builds the forward pass for a loaded model file, by its architecture.

use std::sync::Arc;

use anyhow::{Context, Result, bail};

use crate::engine::arch::ModelForward;
use crate::engine::arch::bailingmoe::BailingMoeModel;
use crate::engine::arch::deepseek4::Deepseek4Model;
use crate::engine::arch::deepseek41::Deepseek41Model;
use crate::engine::arch::dflash::DFlashModel;
use crate::engine::arch::gemma::GemmaModel;
use crate::engine::arch::glm5::Glm5Model;
use crate::engine::arch::inkling::InklingModel;
use crate::engine::arch::inkling2::Inkling2Model;
use crate::engine::arch::kimi3::Kimi3Model;
use crate::engine::arch::llama::LlamaModel;
use crate::engine::arch::mistral::MistralModel;
use crate::engine::arch::muse::MuseModel;
use crate::engine::arch::muse_spark::MuseSparkModel;
use crate::engine::arch::nemotron::NemotronModel;
use crate::engine::arch::phi::PhiModel;
use crate::engine::arch::qwen3next::Qwen3NextModel;
use crate::engine::arch::qwen4exp::Qwen4ExpModel;
use crate::engine::arch::qwen35::Qwen35Model;
use crate::engine::arch::qwen35moe::Qwen35MoeModel;
use crate::engine::backend::Backend;
use crate::engine::loader::{ArchFamily, LoadedModel};

/// The forward pass for `loaded`, running its matmuls on `backend`.
pub fn model(loaded: &LoadedModel, backend: &Arc<dyn Backend>) -> Result<Arc<dyn ModelForward>> {
    let architecture = loaded.config.architecture.clone();
    let family = crate::engine::loader::resolve_arch_family(&architecture)?;
    // A Hadamard-folded file (`crate::engine::hadamard`) is only served by the
    // architectures whose loaders route every folded weight through the
    // fold ledger. The rest would load it as if the weights were plain and
    // generate confidently wrong text, so they refuse it up front.
    if !matches!(
        family,
        ArchFamily::Qwen35 | ArchFamily::Qwen35Moe | ArchFamily::Qwen3Next | ArchFamily::Qwen4Exp
    ) && crate::engine::hadamard::from_loaded(loaded)?.is_some()
    {
        bail!(
            "this file's weights are Hadamard-folded (prism.hadamard.*), and the `{architecture}`              architecture does not apply the activation transform that requires; only the Qwen              3.5-family architectures do"
        );
    }
    let model: Arc<dyn ModelForward> = match family {
        ArchFamily::LlamaStyle => Arc::new(
            LlamaModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Gemma => Arc::new(
            GemmaModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Qwen35Moe => Arc::new(
            Qwen35MoeModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Qwen35 => Arc::new(
            Qwen35Model::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Qwen3Next => Arc::new(
            Qwen3NextModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Qwen4Exp => Arc::new(
            Qwen4ExpModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::DFlash => {
            Arc::new(DFlashModel::load_with_backend(loaded).context("building model")?)
        }
        ArchFamily::Deepseek41 => Arc::new(
            Deepseek41Model::load_with_backend(loaded, backend.clone())
                .context("building model")?,
        ),
        ArchFamily::Deepseek4 => Arc::new(
            Deepseek4Model::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        // `glm-dsa` and `mistral-large4` are two configurations of one
        // trunk, `arch::dsa`; each module reads its own file conventions
        // into it.
        ArchFamily::GlmDsa => Arc::new(
            crate::engine::arch::glm::load_with_backend(loaded, backend.clone())
                .context("building model")?,
        ),
        ArchFamily::MistralLarge4 => Arc::new(
            crate::engine::arch::mistral_large4::load_with_backend(loaded, backend.clone())
                .context("building model")?,
        ),
        ArchFamily::MinimaxM3 => Arc::new(
            crate::engine::arch::minimax_m3::MinimaxM3Model::load_with_backend(
                loaded,
                backend.clone(),
            )
            .context("building model")?,
        ),
        ArchFamily::Glm5Next => Arc::new(
            Glm5Model::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::KimiK3 => Arc::new(
            Kimi3Model::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Phi3 => Arc::new(
            PhiModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Mistral3 => Arc::new(
            MistralModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        // `arch::granite` returns a `LlamaModel` — Granite's block *is*
        // llama's, and the module is the four multipliers that make it
        // Granite rather than a second forward pass.
        ArchFamily::Granite => Arc::new(
            crate::engine::arch::granite::load_with_backend(loaded, backend.clone())
                .context("building model")?,
        ),
        ArchFamily::Muse => Arc::new(
            MuseModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::MuseSpark => Arc::new(
            MuseSparkModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Inkling => Arc::new(
            InklingModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Inkling2 => Arc::new(
            Inkling2Model::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::NemotronHMoe => Arc::new(
            NemotronModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::BailingMoe3 => Arc::new(
            BailingMoeModel::load_with_backend(loaded, backend.clone())
                .context("building model")?,
        ),
        // Not a language model: `prepare` builds the served `ModelForward`
        // from its text encoder and the transformer through
        // `crate::engine::image::Pipeline`, and never reaches here with one. A
        // draft-model spec naming one is the only way in.
        ArchFamily::QwenImage => bail!(
            "{architecture} is a diffusion transformer, not a language model; it is served as \
             the model, with its text encoder and VAE beside it, and cannot stand in as a draft"
        ),
        ArchFamily::Qwen3Tts => bail!(
            "{architecture} is a speech generator, not a language model; it is served as the \
             model, with its codec and voices beside it, and cannot stand in as a draft"
        ),
        ArchFamily::Ltxv => bail!(
            "{architecture} is a video diffusion transformer, not a language model; it is \
             served as the model, with its text encoder and VAEs beside it, and cannot stand in \
             as a draft"
        ),
    };
    Ok(model)
}
