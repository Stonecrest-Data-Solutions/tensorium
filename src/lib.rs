pub mod tensor_objects;
pub mod tensor_ops;

#[cfg(test)]
mod tests;

pub use tensor_objects::{
    Tensor,
    TensorIndexResult
};