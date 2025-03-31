//! # Broadcasting
//!
//! Broadcasting is a term that is used to describe the action of expanding tensor sizes into
//! compatible shapes. This is an important system for a numerical math library as we often need to
//! perform operations with tensors that aren't the same shape. tensorium uses __numpy__ style
//! broadcasting which is linked [here](https://numpy.org/doc/stable/user/basics.broadcasting.html).
//!
//! ## Broadcasting Purpose
//!
//! When running operations on Tensors, it is often found that two Tensor's shapes don't exactly
//! match. In the simplest example, a Tensor of shape \[2, 2\] might need to be multiplied by a
//! scalar Tensor of shape \[1\]. To successfully implement this multiplication, we must ensure that
//! the Tensors have the same shape. Therefor the latter Tensor must be expanded into shape \[2, 2\]
//! by copying its scalar value across dimensions.
//!
//! Much more complex scenarios exist, such as broadcasting a shape \[4, 1, 3\] Tensor into a shape
//! \[5, 4, 5, 3\] Tensor. The functions in this module are designed to make broadcasting as
//! painless as possible for the users. The general rules for broadcasting are given below, but we
//! suggest ready Numpy's article for a deeper explanation.
//!
//! ## General Broadcasting Rules
//!
//! When two Tensors of different shapes are to be broadcast together, their shapes are compared
//! element-wise, starting from the trailing (i.e. rightmost) dimension and moving toward the
//! leading (i.e. leftmost) element. If the two shapes are of differing dimensions, then the shorter
//! shape is padded with leading ones until it matches the dimensionality of the longer shape. For
//! each element-wise comparison, one of two statements __must__ be true for the shapes to be
//! broadcastable.
//!
//! 1. They shapes are equal
//! 2. One of them is 1
//!
//! If neither of these are true for a single comparison, the shapes are not broadcastable and have
//! incompatible shapes.
//!
//! During broadcasting, if the value of a dimension between the two Tensors is equal, nothing
//! happens and the next dimension is considered. If one of the dimensions is one, then that Tensor
//! has that dimension copied (or stretched) until that dimension equals that dimension on the other
//! Tensor.
//!
//! ## Example
//!
//! Below is a full example of a realistic broadcasting scenario that you may encounter. There are a
//! few involved steps and this shows it in its entirety. It shows how one might broadcast a 3x1 and
//! a 1x3 Tensors into 3x3 Tensors so they can be multiplied together.
//!
//! ```
//! use tensorium::Tensor;
//! use tensorium::tensor_ops;
//!
//! // Initialize our Tensors
//! let t1 = Tensor::Element(vec![1.0, 2.0, 3.0]);
//! let t2 = Tensor::Array(Vec::from([
//!     Tensor::Element(vec![1.0]),
//!     Tensor::Element(vec![2.0]),
//!     Tensor::Element(vec![3.0])
//! ]));
//!
//! // First we need their shapes
//! let t1_shape = tensor_ops::get_dimension(&t1);
//! let t2_shape = tensor_ops::get_dimension(&t2);
//!
//! // Now get the final broadcast shape. This is the shape the two tensors will be broadcast to.
//! let broadcast_shape = tensor_ops::broadcast_shape(&t1_shape, &t2_shape);
//!
//! // Pad each tensor with extra dimensions. t2 here will remain unchanged.
//! let t1 = tensor_ops::expand_dims(t1, broadcast_shape.len() - t1_shape.len());
//! let t2 = tensor_ops::expand_dims(t2, broadcast_shape.len() - t2_shape.len());
//!
//! // Broadcast our Tensors. They are now each 3x3 Tensors
//! let t1_b = tensor_ops::broadcast_tensor(&t1, &broadcast_shape);
//! let t2_b = tensor_ops::broadcast_tensor(&t2, &broadcast_shape);
//!
//! // Multiply our tensors together
//! let t12 = tensor_ops::multiply_tensors(&t1_b, &t2_b);
//!
//! // Check our work
//! let truth_tensor = Tensor::Array(Vec::from([
//!     Tensor::Element(vec![1.0, 2.0, 3.0]),
//!     Tensor::Element(vec![2.0, 4.0, 6.0]),
//!     Tensor::Element(vec![3.0, 6.0, 9.0])
//! ]));
//!
//! assert_eq!(t12, truth_tensor);
//! ```
//!
//!
mod standard_ops;
pub use standard_ops::{
    add_tensors,
    subtract_tensors,
    multiply_tensors,
    divide_tensors,
    remainder_tensors,
    tensor_op
};

mod utilities;
pub use utilities::{
    get_dimension
};

mod tensor_creation;
pub use tensor_creation::{
    zero_tensor
};

mod broadcasting;
pub use broadcasting::{
    is_broadcastable,
    broadcast_shape,
    expand_tensor,
    broadcast_tensor,
    expand_dims,
};
