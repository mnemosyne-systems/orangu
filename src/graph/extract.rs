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

use std::path::Path;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Node, Parser, Query, QueryCursor};

// ── Public data types ─────────────────────────────────────────────────────────

use serde::{Deserialize, Serialize};

/// Confidence level of an extracted relationship.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Confidence {
    /// Concrete syntactic relationship (call expression, `use` statement, etc.)
    Extracted,
    /// Structural / heuristic guess (e.g. external call whose callee is unknown).
    Inferred,
}

/// A symbol node extracted from source code.
#[derive(Debug, Clone)]
pub struct ExtractedNode {
    pub id: String,
    pub label: String,
    pub source_file: String,
    pub source_location: String,
    pub kind: String,
}

/// A directed relationship between two nodes.
#[derive(Debug, Clone)]
pub struct ExtractedEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub confidence: Confidence,
}

// ── Language-specific query strings ──────────────────────────────────────────

mod rust_queries {
    pub const DEFS: &str = r#"
      (function_item    name: (identifier)      @name) @fn
      (struct_item      name: (type_identifier) @name) @struct
      (enum_item        name: (type_identifier) @name) @enum
      (trait_item       name: (type_identifier) @name) @trait
      (mod_item         name: (identifier)      @name) @module
      (type_item        name: (type_identifier) @name) @type_alias
      (union_item       name: (type_identifier) @name) @union
      (macro_definition name: (identifier)      @name) @macro
      (impl_item        type: (type_identifier) @name) @impl
    "#;

    pub const CALLS: &str = r#"
      (call_expression function: (identifier) @callee) @call
      (call_expression function: (field_expression field: (field_identifier) @callee)) @call
      (macro_invocation macro: (identifier) @callee) @call
    "#;

    pub const IMPORTS: &str = r#"
      (use_declaration argument: (_) @import_path)
      (extern_crate_declaration name: (identifier) @import_path)
    "#;
}

mod python_queries {
    pub const DEFS: &str = r#"
      (function_definition name: (identifier) @name) @fn
      (class_definition    name: (identifier) @name) @class
    "#;

    pub const CALLS: &str = r#"
      (call function: (identifier) @callee) @call
      (call function: (attribute attribute: (identifier) @callee)) @call
    "#;

    pub const IMPORTS: &str = r#"
      (import_statement name: (dotted_name) @import_path)
      (import_from_statement module_name: (dotted_name) @import_path)
    "#;
}

mod js_queries {
    pub const DEFS: &str = r#"
      (function_declaration name: (identifier) @name) @fn
      (class_declaration    name: (identifier) @name) @class
      (method_definition    name: (property_identifier) @name) @method
      (arrow_function) @fn
    "#;

    pub const CALLS: &str = r#"
      (call_expression function: (identifier) @callee) @call
      (call_expression function: (member_expression property: (property_identifier) @callee)) @call
    "#;

    pub const IMPORTS: &str = r#"
      (import_statement source: (string) @import_path)
    "#;
}

mod ts_queries {
    pub const DEFS: &str = r#"
      (function_declaration name: (identifier) @name) @fn
      (class_declaration    name: (type_identifier) @name) @class
      (method_definition    name: (property_identifier) @name) @method
      (arrow_function) @fn
      (interface_declaration name: (type_identifier) @name) @interface
      (type_alias_declaration name: (type_identifier) @name) @type
      (enum_declaration name: (identifier) @name) @enum
    "#;

    pub const CALLS: &str = r#"
      (call_expression function: (identifier) @callee) @call
      (call_expression function: (member_expression property: (property_identifier) @callee)) @call
    "#;

    pub const IMPORTS: &str = r#"
      (import_statement source: (string) @import_path)
    "#;
}

mod tsx_queries {
    pub const DEFS: &str = r#"
      (function_declaration name: (identifier) @name) @fn
      (class_declaration    name: (type_identifier) @name) @class
      (method_definition    name: (property_identifier) @name) @method
      (arrow_function) @fn
      (interface_declaration name: (type_identifier) @name) @interface
      (type_alias_declaration name: (type_identifier) @name) @type
      (enum_declaration name: (identifier) @name) @enum
    "#;

    pub const CALLS: &str = r#"
      (call_expression function: (identifier) @callee) @call
      (call_expression function: (member_expression property: (property_identifier) @callee)) @call
      (jsx_element open_tag: (jsx_opening_element name: (identifier) @callee)) @call
      (jsx_self_closing_element name: (identifier) @callee) @call
    "#;

    pub const IMPORTS: &str = r#"
      (import_statement source: (string) @import_path)
    "#;
}

mod go_queries {
    pub const DEFS: &str = r#"
      (function_declaration  name: (identifier)        @name) @fn
      (method_declaration    name: (field_identifier)  @name) @method
      (type_declaration (type_spec name: (type_identifier) @name)) @type
    "#;

    pub const CALLS: &str = r#"
      (call_expression function: (identifier) @callee) @call
      (call_expression function: (selector_expression field: (field_identifier) @callee)) @call
    "#;

    pub const IMPORTS: &str = r#"
      (import_spec path: (interpreted_string_literal) @import_path)
    "#;
}

mod java_queries {
    pub const DEFS: &str = r#"
      (method_declaration name: (identifier) @name) @method
      (class_declaration  name: (identifier) @name) @class
      (interface_declaration name: (identifier) @name) @interface
      (enum_declaration name: (identifier) @name) @enum
    "#;

    pub const CALLS: &str = r#"
      (method_invocation name: (identifier) @callee) @call
    "#;

    pub const IMPORTS: &str = r#"
      (import_declaration (scoped_identifier) @import_path)
    "#;
}

mod cpp_queries {
    pub const DEFS: &str = r#"
      (function_definition declarator: (function_declarator declarator: (identifier) @name)) @fn
      (function_definition declarator: (function_declarator declarator: (qualified_identifier name: (identifier) @name))) @method
      (struct_specifier    name: (type_identifier) @name) @struct
      (class_specifier     name: (type_identifier) @name) @class
      (enum_specifier      name: (type_identifier) @name) @enum
    "#;

    pub const CALLS: &str = r#"
      (call_expression function: (identifier) @callee) @call
      (call_expression function: (field_expression field: (field_identifier) @callee)) @call
    "#;

    pub const IMPORTS: &str = r#"
      (preproc_include path: (system_lib_string) @import_path)
    "#;
}

mod c_queries {
    pub const DEFS: &str = r#"
      (function_definition declarator: (function_declarator declarator: (identifier) @name)) @fn
      (struct_specifier    name: (type_identifier) @name) @struct
      (enum_specifier      name: (type_identifier) @name) @enum
    "#;

    pub const CALLS: &str = r#"
      (call_expression function: (identifier) @callee) @call
    "#;

    pub const IMPORTS: &str = r#"
      (preproc_include path: (string_literal) @import_path)
      (preproc_include path: (system_lib_string) @import_path)
    "#;
}

mod csharp_queries {
    pub const DEFS: &str = r#"
      (class_declaration name: (identifier) @name) @class
      (struct_declaration name: (identifier) @name) @struct
      (interface_declaration name: (identifier) @name) @interface
      (enum_declaration name: (identifier) @name) @enum
      (method_declaration name: (identifier) @name) @method
      (namespace_declaration name: (identifier) @name) @module
    "#;

    pub const CALLS: &str = r#"
      (invocation_expression function: (identifier) @callee) @call
      (invocation_expression function: (member_access_expression name: (identifier) @callee)) @call
    "#;

    pub const IMPORTS: &str = r#"
      (using_directive (identifier) @import_path)
    "#;
}

mod ruby_queries {
    pub const DEFS: &str = r#"
      (class name: (constant) @name) @class
      (module name: (constant) @name) @module
      (method name: (identifier) @name) @fn
      (singleton_method name: (identifier) @name) @fn
    "#;

    pub const CALLS: &str = r#"
      (call method: (identifier) @callee) @call
    "#;

    pub const IMPORTS: &str = "";
}

mod bash_queries {
    pub const DEFS: &str = r#"
      (function_definition name: (word) @name) @fn
    "#;

    pub const CALLS: &str = r#"
      (command name: (command_name (word) @callee)) @call
    "#;

    pub const IMPORTS: &str = "";
}

mod scala_queries {
    pub const DEFS: &str = r#"
      (class_definition name: (identifier) @name) @class
      (object_definition name: (identifier) @name) @module
      (trait_definition name: (identifier) @name) @trait
      (function_definition name: (identifier) @name) @fn
      (function_declaration name: (identifier) @name) @fn
    "#;

    pub const CALLS: &str = r#"
      (call_expression function: (identifier) @callee) @call
    "#;

    pub const IMPORTS: &str = r#"
      (import_declaration path: (identifier) @import_path)
    "#;
}

mod julia_queries {
    pub const DEFS: &str = r#"
      (function_definition (signature (call_expression (identifier) @name))) @fn
      (struct_definition (type_head (identifier) @name)) @struct
      (struct_definition (type_head (binary_expression (identifier) @name))) @struct
      (module_definition name: (identifier) @name) @module
      (macro_definition (signature (call_expression (identifier) @name))) @macro
      (abstract_definition (type_head (identifier) @name)) @type
    "#;

    pub const CALLS: &str = r#"
      (call_expression (identifier) @callee) @call
    "#;

    pub const IMPORTS: &str = r#"
      (using_statement (identifier) @import_path)
      (import_statement (selected_import (identifier) @import_path))
    "#;
}

mod lua_queries {
    pub const DEFS: &str = r#"
      (function_declaration name: (identifier) @name) @fn
      (function_declaration name: (dot_index_expression field: (identifier) @name)) @fn
      (function_declaration name: (method_index_expression method: (identifier) @name)) @method
    "#;

    pub const CALLS: &str = r#"
      (function_call name: (identifier) @callee) @call
      (function_call name: (dot_index_expression field: (identifier) @callee)) @call
      (function_call name: (method_index_expression method: (identifier) @callee)) @call
    "#;

    pub const IMPORTS: &str = "";
}

mod r_queries {
    pub const DEFS: &str = r#"
      (binary_operator lhs: (identifier) @name rhs: (function_definition)) @fn
    "#;

    pub const CALLS: &str = r#"
      (call function: (identifier) @callee) @call
    "#;

    pub const IMPORTS: &str = "";
}

mod zig_queries {
    pub const DEFS: &str = r#"
      (function_declaration name: (identifier) @name) @fn
      (variable_declaration (identifier) @name (struct_declaration)) @struct
      (variable_declaration (identifier) @name (enum_declaration)) @enum
    "#;

    pub const CALLS: &str = r#"
      (call_expression function: (identifier) @callee) @call
      (call_expression function: (field_expression member: (identifier) @callee)) @call
    "#;

    pub const IMPORTS: &str = r#"
      (builtin_function (builtin_identifier) (arguments (string (string_content) @import_path)))
    "#;
}

mod swift_queries {
    pub const DEFS: &str = r#"
      (function_declaration name: (simple_identifier) @name) @fn
      (class_declaration name: (type_identifier) @name) @class
      (protocol_declaration name: (type_identifier) @name) @interface
    "#;

    pub const CALLS: &str = r#"
      (call_expression (simple_identifier) @callee) @call
    "#;

    pub const IMPORTS: &str = r#"
      (import_declaration (identifier) @import_path)
    "#;
}

mod dart_queries {
    pub const DEFS: &str = r#"
      (class_declaration name: (identifier) @name) @class
      (mixin_declaration name: (identifier) @name) @class
      (enum_declaration name: (identifier) @name) @enum
      (function_declaration signature: (function_signature name: (identifier) @name)) @fn
      (method_declaration signature: (method_signature (function_signature name: (identifier) @name))) @method
    "#;

    pub const CALLS: &str = r#"
      (call_expression function: (identifier) @callee) @call
      (call_expression function: (member_expression property: (identifier) @callee)) @call
    "#;

    pub const IMPORTS: &str = r#"
      (import_or_export (library_import (import_specification uri: (configurable_uri (uri (string_literal) @import_path)))))
    "#;
}

mod erlang_queries {
    pub const DEFS: &str = r#"
      (fun_decl clause: (function_clause name: (atom) @name)) @fn
    "#;

    pub const CALLS: &str = r#"
      (call expr: (atom) @callee) @call
    "#;

    pub const IMPORTS: &str = "";
}

mod php_queries {
    pub const DEFS: &str = r#"
      (class_declaration name: (name) @name) @class
      (interface_declaration name: (name) @name) @interface
      (trait_declaration name: (name) @name) @trait
      (method_declaration name: (name) @name) @fn
      (function_definition name: (name) @name) @fn
    "#;

    pub const CALLS: &str = r#"
      (function_call_expression function: (name) @callee) @call
      (member_call_expression name: (name) @callee) @call
      (scoped_call_expression name: (name) @callee) @call
    "#;

    pub const IMPORTS: &str = r#"
      (namespace_use_declaration (namespace_use_clause (qualified_name) @import_path))
      (namespace_use_declaration (namespace_use_clause (name) @import_path))
      (require_expression [(string (string_content) @import_path) (encapsed_string (string_content) @import_path)])
      (require_once_expression [(string (string_content) @import_path) (encapsed_string (string_content) @import_path)])
      (include_expression [(string (string_content) @import_path) (encapsed_string (string_content) @import_path)])
      (include_once_expression [(string (string_content) @import_path) (encapsed_string (string_content) @import_path)])
    "#;
}

mod ocaml_queries {
    pub const DEFS: &str = r#"
      (module_definition (module_binding (module_name) @name)) @module
      (value_definition (let_binding pattern: (value_name) @name)) @fn
      (type_definition (type_binding name: (type_constructor) @name)) @type
      (exception_definition (constructor_declaration (constructor_name) @name)) @class
      (class_definition (class_binding (class_name) @name)) @class
      (method_definition (method_name) @name) @fn
      (module_type_definition (module_type_name) @name) @interface
    "#;
    pub const CALLS: &str = r#"
      (application_expression function: (value_path (value_name) @callee)) @call
    "#;
    pub const IMPORTS: &str = r#"
      (open_module module: (module_path (module_name) @import_path))
    "#;
}

mod ocaml_interface_queries {
    pub const DEFS: &str = r#"
      (module_definition (module_binding (module_name) @name)) @module
      (value_specification (value_name) @name) @fn
      (type_definition (type_binding name: (type_constructor) @name)) @type
      (exception_definition (constructor_declaration (constructor_name) @name)) @class
      (class_definition (class_binding (class_name) @name)) @class
      (method_specification (method_name) @name) @fn
      (module_type_definition (module_type_name) @name) @interface
    "#;
    pub const CALLS: &str = "";
    pub const IMPORTS: &str = r#"
      (open_module module: (module_path (module_name) @import_path))
    "#;
}

// ── Language enum ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportedLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    Go,
    Java,
    Cpp,
    C,
    CSharp,
    Ruby,
    Bash,
    Scala,
    Julia,
    Lua,
    R,
    Zig,
    Swift,
    Dart,
    Erlang,
    Php,
    Ocaml,
    OcamlInterface,
}

impl SupportedLanguage {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(Self::Rust),
            "py" => Some(Self::Python),
            "js" | "mjs" | "cjs" => Some(Self::JavaScript),
            "ts" => Some(Self::TypeScript),
            "tsx" => Some(Self::Tsx),
            "go" => Some(Self::Go),
            "java" => Some(Self::Java),
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Some(Self::Cpp),
            "c" | "h" => Some(Self::C),
            "cs" => Some(Self::CSharp),
            "rb" => Some(Self::Ruby),
            "sh" | "bash" => Some(Self::Bash),
            "scala" | "sc" => Some(Self::Scala),
            "jl" => Some(Self::Julia),
            "lua" => Some(Self::Lua),
            "r" | "R" => Some(Self::R),
            "zig" => Some(Self::Zig),
            "swift" => Some(Self::Swift),
            "dart" => Some(Self::Dart),
            "erl" | "hrl" => Some(Self::Erlang),
            "php" => Some(Self::Php),
            "ml" => Some(Self::Ocaml),
            "mli" => Some(Self::OcamlInterface),
            _ => None,
        }
    }

    fn ts_language(self) -> Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Self::Go => tree_sitter_go::LANGUAGE.into(),
            Self::Java => tree_sitter_java::LANGUAGE.into(),
            Self::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Self::C => tree_sitter_c::LANGUAGE.into(),
            Self::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
            Self::Ruby => tree_sitter_ruby::LANGUAGE.into(),
            Self::Bash => tree_sitter_bash::LANGUAGE.into(),
            Self::Scala => tree_sitter_scala::LANGUAGE.into(),
            Self::Julia => tree_sitter_julia::LANGUAGE.into(),
            Self::Lua => tree_sitter_lua::LANGUAGE.into(),
            Self::R => tree_sitter_r::LANGUAGE.into(),
            Self::Zig => tree_sitter_zig::LANGUAGE.into(),
            Self::Swift => tree_sitter_swift::LANGUAGE.into(),
            Self::Dart => tree_sitter_dart::LANGUAGE.into(),
            Self::Erlang => tree_sitter_erlang::LANGUAGE.into(),
            Self::Php => tree_sitter_php::LANGUAGE_PHP.into(),
            Self::Ocaml => tree_sitter_ocaml::LANGUAGE_OCAML.into(),
            Self::OcamlInterface => tree_sitter_ocaml::LANGUAGE_OCAML_INTERFACE.into(),
        }
    }

    fn defs_query_str(self) -> &'static str {
        match self {
            Self::Rust => rust_queries::DEFS,
            Self::Python => python_queries::DEFS,
            Self::JavaScript => js_queries::DEFS,
            Self::TypeScript => ts_queries::DEFS,
            Self::Tsx => tsx_queries::DEFS,
            Self::Go => go_queries::DEFS,
            Self::Java => java_queries::DEFS,
            Self::Cpp => cpp_queries::DEFS,
            Self::C => c_queries::DEFS,
            Self::CSharp => csharp_queries::DEFS,
            Self::Ruby => ruby_queries::DEFS,
            Self::Bash => bash_queries::DEFS,
            Self::Scala => scala_queries::DEFS,
            Self::Julia => julia_queries::DEFS,
            Self::Lua => lua_queries::DEFS,
            Self::R => r_queries::DEFS,
            Self::Zig => zig_queries::DEFS,
            Self::Swift => swift_queries::DEFS,
            Self::Dart => dart_queries::DEFS,
            Self::Erlang => erlang_queries::DEFS,
            Self::Php => php_queries::DEFS,
            Self::Ocaml => ocaml_queries::DEFS,
            Self::OcamlInterface => ocaml_interface_queries::DEFS,
        }
    }

    fn calls_query_str(self) -> &'static str {
        match self {
            Self::Rust => rust_queries::CALLS,
            Self::Python => python_queries::CALLS,
            Self::JavaScript => js_queries::CALLS,
            Self::TypeScript => ts_queries::CALLS,
            Self::Tsx => tsx_queries::CALLS,
            Self::Go => go_queries::CALLS,
            Self::Java => java_queries::CALLS,
            Self::Cpp => cpp_queries::CALLS,
            Self::C => c_queries::CALLS,
            Self::CSharp => csharp_queries::CALLS,
            Self::Ruby => ruby_queries::CALLS,
            Self::Bash => bash_queries::CALLS,
            Self::Scala => scala_queries::CALLS,
            Self::Julia => julia_queries::CALLS,
            Self::Lua => lua_queries::CALLS,
            Self::R => r_queries::CALLS,
            Self::Zig => zig_queries::CALLS,
            Self::Swift => swift_queries::CALLS,
            Self::Dart => dart_queries::CALLS,
            Self::Erlang => erlang_queries::CALLS,
            Self::Php => php_queries::CALLS,
            Self::Ocaml => ocaml_queries::CALLS,
            Self::OcamlInterface => ocaml_interface_queries::CALLS,
        }
    }

    fn imports_query_str(self) -> &'static str {
        match self {
            Self::Rust => rust_queries::IMPORTS,
            Self::Python => python_queries::IMPORTS,
            Self::JavaScript => js_queries::IMPORTS,
            Self::TypeScript => ts_queries::IMPORTS,
            Self::Tsx => tsx_queries::IMPORTS,
            Self::Go => go_queries::IMPORTS,
            Self::Java => java_queries::IMPORTS,
            Self::Cpp => cpp_queries::IMPORTS,
            Self::C => c_queries::IMPORTS,
            Self::CSharp => csharp_queries::IMPORTS,
            Self::Ruby => ruby_queries::IMPORTS,
            Self::Bash => bash_queries::IMPORTS,
            Self::Scala => scala_queries::IMPORTS,
            Self::Julia => julia_queries::IMPORTS,
            Self::Lua => lua_queries::IMPORTS,
            Self::R => r_queries::IMPORTS,
            Self::Zig => zig_queries::IMPORTS,
            Self::Swift => swift_queries::IMPORTS,
            Self::Dart => dart_queries::IMPORTS,
            Self::Erlang => erlang_queries::IMPORTS,
            Self::Php => php_queries::IMPORTS,
            Self::Ocaml => ocaml_queries::IMPORTS,
            Self::OcamlInterface => ocaml_interface_queries::IMPORTS,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Tsx => "tsx",
            Self::Go => "go",
            Self::Java => "java",
            Self::Cpp => "cpp",
            Self::C => "c",
            Self::CSharp => "csharp",
            Self::Ruby => "ruby",
            Self::Bash => "bash",
            Self::Scala => "scala",
            Self::Julia => "julia",
            Self::Lua => "lua",
            Self::R => "r",
            Self::Zig => "zig",
            Self::Swift => "swift",
            Self::Dart => "dart",
            Self::Erlang => "erlang",
            Self::Php => "php",
            Self::Ocaml => "ocaml",
            Self::OcamlInterface => "ocaml_interface",
        }
    }
}

// ── Per-language compiled queries ─────────────────────────────────────────────

struct LanguageQueries {
    defs: Query,
    calls: Query,
    imports: Query,
}

impl LanguageQueries {
    fn compile(lang: SupportedLanguage) -> anyhow::Result<Self> {
        let ts_lang = lang.ts_language();
        Ok(Self {
            defs: Query::new(&ts_lang, lang.defs_query_str())
                .map_err(|e| anyhow::anyhow!("bad defs query for {:?}: {e}", lang))?,
            calls: Query::new(&ts_lang, lang.calls_query_str())
                .map_err(|e| anyhow::anyhow!("bad calls query for {:?}: {e}", lang))?,
            imports: Query::new(&ts_lang, lang.imports_query_str())
                .map_err(|e| anyhow::anyhow!("bad imports query for {:?}: {e}", lang))?,
        })
    }
}

// ── Raw intermediate data (owned, borrow-free) ────────────────────────────────

const LANGUAGE_BUILTIN_GLOBALS: &[&str] = &[
    // JavaScript / TypeScript ECMAScript built-ins
    "String",
    "Number",
    "Boolean",
    "Object",
    "Array",
    "Symbol",
    "BigInt",
    "Date",
    "RegExp",
    "Error",
    "TypeError",
    "RangeError",
    "SyntaxError",
    "ReferenceError",
    "EvalError",
    "URIError",
    "Promise",
    "Map",
    "Set",
    "WeakMap",
    "WeakSet",
    "JSON",
    "Math",
    "Reflect",
    "Proxy",
    "Intl",
    "parseInt",
    "parseFloat",
    "isNaN",
    "isFinite",
    "encodeURIComponent",
    "decodeURIComponent",
    "encodeURI",
    "decodeURI",
    // Browser / Node common globals
    "URL",
    "URLSearchParams",
    "FormData",
    "Blob",
    "File",
    "Headers",
    "Request",
    "Response",
    "AbortController",
    "AbortSignal",
    "TextEncoder",
    "TextDecoder",
    "console",
    // Python built-in callables
    "str",
    "int",
    "float",
    "bool",
    "list",
    "dict",
    "set",
    "tuple",
    "bytes",
    "len",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sum",
    "min",
    "max",
    "print",
    "open",
    "isinstance",
    "type",
    "super",
    "sorted",
    "reversed",
    "any",
    "all",
    "abs",
    "round",
    "next",
    "iter",
    "hash",
    "id",
    "repr",
    "callable",
    "getattr",
    "setattr",
    "hasattr",
    "delattr",
    "vars",
    "dir",
    // Rust common built-ins and standard library macros/types
    "String",
    "Vec",
    "Option",
    "Result",
    "Some",
    "None",
    "Ok",
    "Err",
    "Box",
    "Rc",
    "Arc",
    "Mutex",
    "RwLock",
    "Cell",
    "RefCell",
    "format!",
    "print!",
    "println!",
    "eprint!",
    "eprintln!",
    "panic!",
    "assert!",
    "assert_eq!",
    "assert_ne!",
    "unreachable!",
    "todo!",
    "unimplemented!",
    // Go built-ins
    "make",
    "new",
    "append",
    "copy",
    "delete",
    "close",
    "len",
    "cap",
    "panic",
    "recover",
    "print",
    "println",
    "error",
    // Java common built-ins / frequently-seen base classes
    "System",
    "String",
    "Integer",
    "Long",
    "Double",
    "Float",
    "Boolean",
    "Object",
    "Class",
    "Math",
    "StringBuilder",
    "StringBuffer",
    "Exception",
    "RuntimeException",
    "NullPointerException",
    "IllegalArgumentException",
    "Override",
    "SuppressWarnings",
    // C/C++ standard library functions
    "printf",
    "scanf",
    "malloc",
    "calloc",
    "realloc",
    "free",
    "exit",
    "abort",
    "memcpy",
    "memmove",
    "memset",
    "memcmp",
    "strlen",
    "strcpy",
    "strcat",
    "strcmp",
    "std",
    "cout",
    "cin",
    "endl",
    "nullptr",
    "NULL",
    // Ruby built-ins
    "puts",
    "require",
    "require_relative",
    "attr_reader",
    "attr_writer",
    "attr_accessor",
    "raise",
    "lambda",
    "proc",
    "p",
    "pp",
    // Bash built-ins
    "echo",
    "read",
    "test",
    "cd",
    "export",
    "source",
    "local",
    "declare",
    "shift",
    "return",
    "eval",
    "exec",
    // Scala built-ins
    "require",
    // Julia built-ins
    "typeof",
    // Lua built-ins
    "print",
    "type",
    "tostring",
    "tonumber",
    "pairs",
    "ipairs",
    "require",
    "error",
    "pcall",
    "xpcall",
    "setmetatable",
    "getmetatable",
    "rawget",
    "rawset",
    "select",
    "unpack",
    "dofile",
    "loadfile",
    // R built-ins
    "library",
    "cat",
    "paste",
    "paste0",
    "sprintf",
    "stop",
    "warning",
    "message",
    "is.null",
    "is.na",
    "c",
    "seq",
    "rep",
    "nrow",
    "ncol",
    "dim",
    "names",
    "data.frame",
    // Swift built-ins
    "print",
    "fatalError",
    "precondition",
    "preconditionFailure",
    "assert",
    "assertionFailure",
    "debugPrint",
    "dump",
    // Dart built-ins
    "print",
    "debugPrint",
    // Erlang built-ins
    "spawn",
    "register",
    "whereis",
    "send",
    "receive",
    "self",
    "exit",
    "link",
    "monitor",
    "demonitor",
    "node",
    "nodes",
    "halt",
    "apply",
    // PHP built-ins
    "echo",
    "print",
    "isset",
    "empty",
    "strlen",
    "count",
    "is_array",
    "is_string",
    "is_numeric",
    "in_array",
    "array_push",
    "array_pop",
    "array_merge",
    "array_keys",
    "array_values",
    "header",
    "die",
    "exit",
    "var_dump",
    "print_r",
    "json_encode",
    "json_decode",
    // OCaml built-ins
    "print_endline",
    "printf",
    "sprintf",
    "failwith",
    "raise",
    "ignore",
    "fst",
    "snd",
];

struct RawDef {
    sym_name: String,
    kind: String,
    location: String,
}

struct RawCall {
    callee_name: String,
    callee_byte_range: (usize, usize),
}

struct RawImport {
    import_text: String,
}

// ── GraphExtractor ────────────────────────────────────────────────────────────

/// Polyglot AST extractor. Supports Rust, Python, JavaScript/TypeScript, Go, Java, and C/C++.
/// One instance keeps one `Parser` and pre-compiled queries per language.
pub struct GraphExtractor {
    rust: LanguageQueries,
    python: LanguageQueries,
    javascript: LanguageQueries,
    typescript: LanguageQueries,
    tsx: LanguageQueries,
    go: LanguageQueries,
    java: LanguageQueries,
    cpp: LanguageQueries,
    c: LanguageQueries,
    csharp: LanguageQueries,
    ruby: LanguageQueries,
    bash: LanguageQueries,
    scala: LanguageQueries,
    julia: LanguageQueries,
    lua: LanguageQueries,
    r: LanguageQueries,
    zig: LanguageQueries,
    swift: LanguageQueries,
    dart: LanguageQueries,
    erlang: LanguageQueries,
    php: LanguageQueries,
    ocaml: LanguageQueries,
    ocaml_interface: LanguageQueries,
}

impl GraphExtractor {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            rust: LanguageQueries::compile(SupportedLanguage::Rust)?,
            python: LanguageQueries::compile(SupportedLanguage::Python)?,
            javascript: LanguageQueries::compile(SupportedLanguage::JavaScript)?,
            typescript: LanguageQueries::compile(SupportedLanguage::TypeScript)?,
            tsx: LanguageQueries::compile(SupportedLanguage::Tsx)?,
            go: LanguageQueries::compile(SupportedLanguage::Go)?,
            java: LanguageQueries::compile(SupportedLanguage::Java)?,
            cpp: LanguageQueries::compile(SupportedLanguage::Cpp)?,
            c: LanguageQueries::compile(SupportedLanguage::C)?,
            csharp: LanguageQueries::compile(SupportedLanguage::CSharp)?,
            ruby: LanguageQueries::compile(SupportedLanguage::Ruby)?,
            bash: LanguageQueries::compile(SupportedLanguage::Bash)?,
            scala: LanguageQueries::compile(SupportedLanguage::Scala)?,
            julia: LanguageQueries::compile(SupportedLanguage::Julia)?,
            lua: LanguageQueries::compile(SupportedLanguage::Lua)?,
            r: LanguageQueries::compile(SupportedLanguage::R)?,
            zig: LanguageQueries::compile(SupportedLanguage::Zig)?,
            swift: LanguageQueries::compile(SupportedLanguage::Swift)?,
            dart: LanguageQueries::compile(SupportedLanguage::Dart)?,
            erlang: LanguageQueries::compile(SupportedLanguage::Erlang)?,
            php: LanguageQueries::compile(SupportedLanguage::Php)?,
            ocaml: LanguageQueries::compile(SupportedLanguage::Ocaml)?,
            ocaml_interface: LanguageQueries::compile(SupportedLanguage::OcamlInterface)?,
        })
    }

    /// Detects the language from `path`'s extension and extracts nodes + edges.
    /// Returns `Ok(([], []))` for unsupported file types rather than an error.
    pub fn extract_from_file(
        &self,
        path: &Path,
        rel_path: &str,
        content: &str,
    ) -> anyhow::Result<(Vec<ExtractedNode>, Vec<ExtractedEdge>)> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let Some(lang) = SupportedLanguage::from_extension(ext) else {
            return Ok((vec![], vec![]));
        };

        let mut parser = Parser::new();
        parser
            .set_language(&lang.ts_language())
            .map_err(|e| anyhow::anyhow!("Failed to set language: {e}"))?;

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| anyhow::anyhow!("tree-sitter failed to parse {:?}", path))?;

        let root = tree.root_node();
        let bytes = content.as_bytes();
        let queries = self.queries(lang);

        let file_stem = Path::new(rel_path)
            .with_extension("")
            .to_string_lossy()
            .replace([std::path::MAIN_SEPARATOR, '/'], "_");
        let file_str = rel_path.to_string();

        // Pass 1: definitions
        let raw_defs: Vec<RawDef> = {
            let mut cursor = QueryCursor::new();
            let mut matches = cursor.matches(&queries.defs, root, bytes);
            let mut acc = Vec::new();
            while let Some(m) = matches.next() {
                let name_cap = m
                    .captures
                    .iter()
                    .find(|c| queries.defs.capture_names()[c.index as usize] == "name");
                let def_cap = m
                    .captures
                    .iter()
                    .find(|c| queries.defs.capture_names()[c.index as usize] != "name");
                if let (Some(nc), Some(dc)) = (name_cap, def_cap) {
                    acc.push(RawDef {
                        sym_name: node_text(nc.node, bytes),
                        kind: queries.defs.capture_names()[dc.index as usize].to_string(),
                        location: location_str(dc.node),
                    });
                }
            }
            acc
        };

        // Pass 2: calls
        let raw_calls: Vec<RawCall> = {
            let mut cursor = QueryCursor::new();
            let mut matches = cursor.matches(&queries.calls, root, bytes);
            let mut acc = Vec::new();
            while let Some(m) = matches.next() {
                let callee_cap = m
                    .captures
                    .iter()
                    .find(|c| queries.calls.capture_names()[c.index as usize] == "callee");
                if let Some(cap) = callee_cap {
                    acc.push(RawCall {
                        callee_name: node_text(cap.node, bytes),
                        callee_byte_range: (cap.node.start_byte(), cap.node.end_byte()),
                    });
                }
            }
            acc
        };

        // Pass 3: imports
        let raw_imports: Vec<RawImport> = {
            let mut cursor = QueryCursor::new();
            let mut matches = cursor.matches(&queries.imports, root, bytes);
            let mut acc = Vec::new();
            while let Some(m) = matches.next() {
                let import_cap = m
                    .captures
                    .iter()
                    .find(|c| queries.imports.capture_names()[c.index as usize] == "import_path");
                if let Some(cap) = import_cap {
                    acc.push(RawImport {
                        import_text: node_text(cap.node, bytes),
                    });
                }
            }
            acc
        };

        // Build output
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut name_to_id: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        for def in raw_defs {
            if LANGUAGE_BUILTIN_GLOBALS.contains(&def.sym_name.as_str()) {
                continue;
            }
            let id = format!("{}::{}", file_stem, def.sym_name);
            name_to_id.insert(def.sym_name.clone(), id.clone());
            nodes.push(ExtractedNode {
                id,
                label: def.sym_name,
                source_file: file_str.clone(),
                source_location: def.location,
                kind: def.kind,
            });
        }

        for call in raw_calls {
            if LANGUAGE_BUILTIN_GLOBALS.contains(&call.callee_name.as_str()) {
                continue;
            }
            if let Some(callee_node) =
                root.descendant_for_byte_range(call.callee_byte_range.0, call.callee_byte_range.1)
                && let Some(enclosing) = enclosing_function_or_method(callee_node, bytes, lang)
            {
                let caller_id = format!("{}::{}", file_stem, enclosing);
                let is_local = name_to_id.contains_key(&call.callee_name);
                let callee_id = name_to_id
                    .get(&call.callee_name)
                    .cloned()
                    .unwrap_or_else(|| format!("external::{}", call.callee_name));

                edges.push(ExtractedEdge {
                    source: caller_id,
                    target: callee_id,
                    relation: "calls".to_string(),
                    confidence: if is_local {
                        Confidence::Extracted
                    } else {
                        Confidence::Inferred
                    },
                });
            }
        }

        let file_module_id = format!("{}::__module__", file_stem);
        for import in raw_imports {
            edges.push(ExtractedEdge {
                source: file_module_id.clone(),
                target: format!("import::{}", import.import_text),
                relation: "imports".to_string(),
                confidence: Confidence::Extracted,
            });
        }

        Ok((nodes, edges))
    }

    fn queries(&self, lang: SupportedLanguage) -> &LanguageQueries {
        match lang {
            SupportedLanguage::Rust => &self.rust,
            SupportedLanguage::Python => &self.python,
            SupportedLanguage::JavaScript => &self.javascript,
            SupportedLanguage::TypeScript => &self.typescript,
            SupportedLanguage::Tsx => &self.tsx,
            SupportedLanguage::Go => &self.go,
            SupportedLanguage::Java => &self.java,
            SupportedLanguage::Cpp => &self.cpp,
            SupportedLanguage::C => &self.c,
            SupportedLanguage::CSharp => &self.csharp,
            SupportedLanguage::Ruby => &self.ruby,
            SupportedLanguage::Bash => &self.bash,
            SupportedLanguage::Scala => &self.scala,
            SupportedLanguage::Julia => &self.julia,
            SupportedLanguage::Lua => &self.lua,
            SupportedLanguage::R => &self.r,
            SupportedLanguage::Zig => &self.zig,
            SupportedLanguage::Swift => &self.swift,
            SupportedLanguage::Dart => &self.dart,
            SupportedLanguage::Erlang => &self.erlang,
            SupportedLanguage::Php => &self.php,
            SupportedLanguage::Ocaml => &self.ocaml,
            SupportedLanguage::OcamlInterface => &self.ocaml_interface,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn node_text<'a>(node: Node<'a>, src: &'a [u8]) -> String {
    node.utf8_text(src)
        .unwrap_or("<invalid utf8>")
        .trim()
        .to_string()
}

fn location_str(node: Node<'_>) -> String {
    let start = node.start_position();
    let end = node.end_position();
    format!("L{}-L{}", start.row + 1, end.row + 1)
}

/// Walks up the parent chain to find the nearest enclosing named function or
/// class, language-aware. Returns `None` for module-level call sites.
fn enclosing_function_or_method(
    mut node: Node<'_>,
    src: &[u8],
    lang: SupportedLanguage,
) -> Option<String> {
    let fn_kinds: &[&str] = match lang {
        SupportedLanguage::Rust => &["function_item"],
        SupportedLanguage::Python => &["function_definition"],
        SupportedLanguage::JavaScript | SupportedLanguage::TypeScript | SupportedLanguage::Tsx => {
            &[
                "function_declaration",
                "method_definition",
                "arrow_function",
            ]
        }
        SupportedLanguage::Go => &["function_declaration", "method_declaration"],
        SupportedLanguage::Java => &["method_declaration"],
        SupportedLanguage::Cpp | SupportedLanguage::C => &["function_definition"],
        SupportedLanguage::CSharp => &["method_declaration"],
        SupportedLanguage::Ruby => &["method", "singleton_method"],
        SupportedLanguage::Bash => &["function_definition"],
        SupportedLanguage::Scala => &["function_definition"],
        SupportedLanguage::Julia => &["function_definition"],
        SupportedLanguage::Lua => &["function_declaration"],
        SupportedLanguage::R => &["function_definition"],
        SupportedLanguage::Zig => &["function_declaration"],
        SupportedLanguage::Swift => &["function_declaration"],
        SupportedLanguage::Dart => &["function_declaration", "method_declaration"],
        SupportedLanguage::Erlang => &["function_clause"],
        SupportedLanguage::Php => &["function_definition", "method_declaration"],
        SupportedLanguage::Ocaml => &["let_binding"],
        SupportedLanguage::OcamlInterface => &["value_specification"],
    };
    let name_field = match lang {
        SupportedLanguage::Rust
        | SupportedLanguage::Python
        | SupportedLanguage::JavaScript
        | SupportedLanguage::TypeScript
        | SupportedLanguage::Tsx
        | SupportedLanguage::Go
        | SupportedLanguage::Java
        | SupportedLanguage::CSharp
        | SupportedLanguage::Ruby
        | SupportedLanguage::Bash
        | SupportedLanguage::Scala
        | SupportedLanguage::Julia
        | SupportedLanguage::Lua
        | SupportedLanguage::R
        | SupportedLanguage::Zig
        | SupportedLanguage::Swift
        | SupportedLanguage::Dart
        | SupportedLanguage::Erlang
        | SupportedLanguage::Php
        | SupportedLanguage::OcamlInterface => "name",
        SupportedLanguage::Ocaml => "pattern",
        // C/C++ function_definition -> declarator -> function_declarator -> declarator
        SupportedLanguage::Cpp | SupportedLanguage::C => "declarator",
    };
    loop {
        node = node.parent()?;
        if fn_kinds.contains(&node.kind()) {
            let name_node = node.child_by_field_name(name_field);
            return name_node.map(|n| node_text(n, src));
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn extractor() -> GraphExtractor {
        GraphExtractor::new().expect("should create extractor")
    }

    const RUST_SAMPLE: &str = r#"
use std::collections::HashMap;
pub struct Cache { data: HashMap<String, String> }
pub fn fetch(key: &str) -> Option<String> { None }
pub fn process(key: &str) -> Option<String> { fetch(key) }
"#;

    const PYTHON_SAMPLE: &str = r#"
import os

class Cache:
    def get(self, key):
        return None

def fetch(key):
    return None

def process(key):
    return fetch(key)
"#;

    const JS_SAMPLE: &str = r#"
import { readFile } from 'fs';
class Cache {}
function fetch(key) { return null; }
function process(key) { return fetch(key); }
"#;

    #[test]
    fn rust_extracts_struct_and_functions() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.rs"), "a.rs", RUST_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"struct"), "{kinds:?}");
        assert!(kinds.contains(&"fn"), "{kinds:?}");
    }

    #[test]
    fn rust_extracts_call_and_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.rs"), "a.rs", RUST_SAMPLE)
            .unwrap();
        assert!(edges.iter().any(|e| e.relation == "calls"), "no call edge");
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    #[test]
    fn python_extracts_class_and_functions() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.py"), "a.py", PYTHON_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"), "{kinds:?}");
        assert!(kinds.contains(&"fn"), "{kinds:?}");
    }

    #[test]
    fn python_extracts_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.py"), "a.py", PYTHON_SAMPLE)
            .unwrap();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    #[test]
    fn javascript_extracts_class_and_functions() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.js"), "a.js", JS_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"), "{kinds:?}");
        assert!(kinds.contains(&"fn"), "{kinds:?}");
    }

    #[test]
    fn javascript_extracts_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.js"), "a.js", JS_SAMPLE)
            .unwrap();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    #[test]
    fn unsupported_extension_returns_empty() {
        let ex = extractor();
        let (nodes, edges) = ex
            .extract_from_file(&PathBuf::from("a.md"), "a.md", "# hello")
            .unwrap();
        assert!(nodes.is_empty());
        assert!(edges.is_empty());
    }

    const PHP_SAMPLE: &str = r#"
    <?php
    require_once 'vendor/autoload.php';
    class User {
        public function getName() {}
    }
    function main() {
        $u = new User();
        $u->getName();
    }
    "#;

    #[test]
    fn php_extracts_class_and_methods() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.php"), "a.php", PHP_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"), "{kinds:?}");
        assert!(kinds.contains(&"fn"), "{kinds:?}");
    }

    #[test]
    fn php_extracts_call_and_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.php"), "a.php", PHP_SAMPLE)
            .unwrap();
        assert!(edges.iter().any(|e| e.relation == "calls"), "no call edge");
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    const OCAML_SAMPLE: &str = r#"
    module M = struct
      let x = 0
      let my_func a = a + 1
    end
    open Core
    let () = print_endline "hello"
    let () = M.my_func 1
    type my_type = int * string
    exception MyError
    class point = object
      val mutable x = 0
      method get_x = x
    end
    module type S = sig end
    "#;

    const OCAML_INTERFACE_SAMPLE: &str = r#"
    module M : sig
      val x : int
      val my_func : int -> int
    end
    open Core
    type my_type = int * string
    exception MyError
    class point : object
      method get_x : int
    end
    module type S = sig end
    "#;

    #[test]
    fn ocaml_extracts_module_and_functions_and_edge_cases() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.ml"), "a.ml", OCAML_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"module"), "{kinds:?}");
        assert!(kinds.contains(&"fn"), "{kinds:?}");
        assert!(kinds.contains(&"type"), "{kinds:?}");
        assert!(kinds.contains(&"class"), "{kinds:?}");
        assert!(kinds.contains(&"interface"), "{kinds:?}");
    }

    #[test]
    fn ocaml_extracts_call_and_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.ml"), "a.ml", OCAML_SAMPLE)
            .unwrap();
        assert!(edges.iter().any(|e| e.relation == "calls"), "no call edge");
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    #[test]
    fn ocaml_interface_extracts_module_and_functions_and_edge_cases() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.mli"), "a.mli", OCAML_INTERFACE_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"module"), "{kinds:?}");
        assert!(kinds.contains(&"fn"), "{kinds:?}");
        assert!(kinds.contains(&"type"), "{kinds:?}");
        assert!(kinds.contains(&"class"), "{kinds:?}");
        assert!(kinds.contains(&"interface"), "{kinds:?}");
    }

    #[test]
    fn ocaml_interface_extracts_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.mli"), "a.mli", OCAML_INTERFACE_SAMPLE)
            .unwrap();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    const GO_SAMPLE: &str = r#"
package main

import "fmt"

type Cache struct{}

func fetch(key string) string { return "" }
func process(key string) string { return fetch(key) }
func (c *Cache) Get(key string) string { return fetch(key) }
"#;

    const JAVA_SAMPLE: &str = r#"
import java.util.List;

public class Cache {
    public String get(String key) { return null; }
    public String process(String key) { return get(key); }
}
"#;

    const CPP_SAMPLE: &str = r#"
#include <string>

struct Cache {};

std::string fetch(std::string key) { return ""; }
std::string process(std::string key) { return fetch(key); }
"#;

    #[test]
    fn go_extracts_functions_and_types() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.go"), "a.go", GO_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(
            kinds.contains(&"fn") || kinds.contains(&"method") || kinds.contains(&"type"),
            "{kinds:?}"
        );
    }

    #[test]
    fn go_extracts_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.go"), "a.go", GO_SAMPLE)
            .unwrap();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    #[test]
    fn java_extracts_class_and_methods() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.java"), "a.java", JAVA_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"), "{kinds:?}");
        assert!(kinds.contains(&"method"), "{kinds:?}");
    }

    #[test]
    fn java_extracts_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.java"), "a.java", JAVA_SAMPLE)
            .unwrap();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    #[test]
    fn cpp_extracts_struct_and_functions() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.cpp"), "a.cpp", CPP_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(
            kinds.contains(&"struct") || kinds.contains(&"fn"),
            "{kinds:?}"
        );
    }

    #[test]
    fn cpp_extracts_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.cpp"), "a.cpp", CPP_SAMPLE)
            .unwrap();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    const CSHARP_SAMPLE: &str = r#"
using System;

public class Cache {
    public string Get(string key) { return Fetch(key); }
    private string Fetch(string key) { return null; }
}
"#;

    #[test]
    fn csharp_extracts_class_and_methods() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.cs"), "a.cs", CSHARP_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"), "{kinds:?}");
        assert!(kinds.contains(&"method"), "{kinds:?}");
    }

    #[test]
    fn csharp_extracts_call_and_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.cs"), "a.cs", CSHARP_SAMPLE)
            .unwrap();
        assert!(edges.iter().any(|e| e.relation == "calls"), "no call edge");
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    const RUBY_SAMPLE: &str = r#"
module MyModule
  class Cache
    def fetch(key)
      get(key)
    end

    def get(key)
      nil
    end
  end
end
"#;

    #[test]
    fn ruby_extracts_class_and_methods() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.rb"), "a.rb", RUBY_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"), "{kinds:?}");
        assert!(kinds.contains(&"fn"), "{kinds:?}");
    }

    const BASH_SAMPLE: &str = r#"#!/bin/bash
fetch() {
    echo "$1"
}

process() {
    fetch "hello"
}
"#;

    #[test]
    fn bash_extracts_functions() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.sh"), "a.sh", BASH_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"fn"), "{kinds:?}");
    }

    const SCALA_SAMPLE: &str = r#"
import scala.collection.mutable

class Cache {
  def fetch(key: String): Option[String] = {
    get(key)
  }

  def get(key: String): Option[String] = None
}
"#;

    #[test]
    fn scala_extracts_class_and_functions() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.scala"), "a.scala", SCALA_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"), "{kinds:?}");
        assert!(kinds.contains(&"fn"), "{kinds:?}");
    }

    #[test]
    fn scala_extracts_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.scala"), "a.scala", SCALA_SAMPLE)
            .unwrap();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    const JULIA_SAMPLE: &str = r#"
using LinearAlgebra

struct Cache
    data::Dict{String, String}
end

function fetch(key::String)
    return get(key)
end
"#;

    #[test]
    fn julia_extracts_struct_and_functions() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.jl"), "a.jl", JULIA_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(
            kinds.contains(&"struct") || kinds.contains(&"fn"),
            "{kinds:?}"
        );
    }

    #[test]
    fn julia_extracts_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.jl"), "a.jl", JULIA_SAMPLE)
            .unwrap();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    const LUA_SAMPLE: &str = r#"
local Cache = {}

function Cache.new()
    return {}
end

function Cache:get(key)
    return self.data[key]
end

local function helper(x)
    return x + 1
end
"#;

    #[test]
    fn lua_extracts_functions() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.lua"), "a.lua", LUA_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(
            kinds.contains(&"fn") || kinds.contains(&"method"),
            "{kinds:?}"
        );
    }

    const R_SAMPLE: &str = r#"
fetch <- function(key) {
  result <- get(key)
  return(result)
}

process <- function(key) {
  fetch(key)
}
"#;

    #[test]
    fn r_extracts_functions() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.r"), "a.r", R_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"fn"), "{kinds:?}");
    }

    const ZIG_SAMPLE: &str = r#"
const std = @import("std");

const Cache = struct {
    pub fn get(self: *Cache, key: []const u8) ?[]const u8 {
        return self.data.get(key);
    }
};

fn helper(x: u32) u32 {
    return x + 1;
}
"#;

    #[test]
    fn zig_extracts_struct_and_functions() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.zig"), "a.zig", ZIG_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(
            kinds.contains(&"struct") || kinds.contains(&"fn"),
            "{kinds:?}"
        );
    }

    #[test]
    fn zig_extracts_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.zig"), "a.zig", ZIG_SAMPLE)
            .unwrap();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    const SWIFT_SAMPLE: &str = r#"
import Foundation

class Cache {
    func get(key: String) -> String? {
        return fetch(key: key)
    }

    func fetch(key: String) -> String? {
        return nil
    }
}
"#;

    #[test]
    fn swift_extracts_class_and_functions() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.swift"), "a.swift", SWIFT_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"), "{kinds:?}");
        assert!(kinds.contains(&"fn"), "{kinds:?}");
    }

    #[test]
    fn swift_extracts_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.swift"), "a.swift", SWIFT_SAMPLE)
            .unwrap();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    const DART_SAMPLE: &str = r#"
import 'dart:async';

class Cache {
  String? get(String key) {
    return fetch(key);
  }

  String? fetch(String key) {
    return null;
  }
}
"#;

    #[test]
    fn dart_extracts_class_and_methods() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.dart"), "a.dart", DART_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"), "{kinds:?}");
    }

    #[test]
    fn dart_extracts_import_edges() {
        let ex = extractor();
        let (_, edges) = ex
            .extract_from_file(&PathBuf::from("a.dart"), "a.dart", DART_SAMPLE)
            .unwrap();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "no import edge"
        );
    }

    const ERLANG_SAMPLE: &str = r#"
-module(cache).
-export([get/1, fetch/1]).

get(Key) ->
    maps:get(Key, #{}).

fetch(Key) ->
    get(Key).
"#;

    #[test]
    fn erlang_extracts_functions() {
        let ex = extractor();
        let (nodes, _) = ex
            .extract_from_file(&PathBuf::from("a.erl"), "a.erl", ERLANG_SAMPLE)
            .unwrap();
        let kinds: Vec<&str> = nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"fn"), "{kinds:?}");
    }
}
