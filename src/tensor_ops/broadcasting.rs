use std::cmp::{ min, max };
use crate:: { Tensor, TensorIndexResult };
use crate::tensor_ops::get_dimension;

/// Determines if two sets of dimensions are broadcastable between each other.
///
/// # Examples
///
/// ```
/// use tensorium::tensor_ops::is_broadcastable;
/// let t1: Vec<usize> = Vec::from([2, 3]);
/// let t2: Vec<usize> = Vec::from([4, 1, 3]);
/// assert!(is_broadcastable(&t1, &t2));
/// ```
///
/// ```
/// use tensorium::tensor_ops::is_broadcastable;
/// let t1: Vec<usize> = Vec::from([2, 3]);
/// let t2: Vec<usize> = Vec::from([1, 3, 3]);
/// assert!(!is_broadcastable(&t1, &t2));
/// ```
pub fn is_broadcastable(ldims: &Vec<usize>, rdims: &Vec<usize>) -> bool {
    if ldims == rdims {
        return true
    }

    // We only need to loop through the short dim's length since it'll be padded with ones later.
    let shorter_dim_length = min(ldims.len(), rdims.len());

    for num in 0..shorter_dim_length {
        let current_ldim = ldims[ldims.len() - num - 1];
        let current_rdim = rdims[rdims.len() - num - 1];

        // If the dims don't match and if neither are 1, then the shapes are not broadcastable.
        if (current_ldim != current_rdim) && (current_ldim != 1) && (current_rdim != 1) {
            return false
        }
    }

    true
}


/// Calculates the final shape that two sizes can be broadcasted to.
///
/// # Examples
///
/// ```
/// use tensorium::tensor_ops::broadcast_shape;
/// let t1: Vec<usize> = Vec::from([3, 2]);
/// let t2: Vec<usize> = Vec::from([4, 1, 2]);
/// let final_shape: Vec<usize> = broadcast_shape(&t1, &t2);
/// assert_eq!(final_shape, Vec::from([4, 3, 2]));
/// ```
pub fn broadcast_shape(ldims: &Vec<usize>, rdims: &Vec<usize>) -> Vec<usize> {
    if !is_broadcastable(ldims, rdims){
        panic!("Shapes are not broadcastable!")
    }

    let longer_dim_length = max(ldims.len(), rdims.len());

    let mut final_dims: Vec<usize> = Vec::new();

    for _ in 0..longer_dim_length {
        final_dims.push(1);
    }

    for num in 0..longer_dim_length {
        let mut current_ldim = 1;
        if ldims.len() > num {
            current_ldim = ldims[ldims.len() - num - 1];
        }

        let mut current_rdim = 1;
        if rdims.len() > num {
            current_rdim = rdims[rdims.len() - num - 1];
        }

        if current_ldim == 1 {
            final_dims[longer_dim_length - num - 1] = current_rdim;
        } else {
            final_dims[longer_dim_length - num - 1] = current_ldim;
        }
    }

    final_dims
}

/// Creates a new tensor of one dimension bigger that is created by copying the given tensor
/// `copy_value` number of times.
///
/// # Examples
///
/// ```
/// use tensorium::Tensor;
/// use tensorium::tensor_ops::expand_tensor;
/// let t1 = Tensor::Element(vec![1.0, 2.0]);
/// let t2 = expand_tensor(&t1, 3);
/// assert_eq!(
///     t2,
///     Tensor::Array(Vec::from([
///         Tensor::Element(vec![1.0, 2.0]),
///         Tensor::Element(vec![1.0, 2.0]),
///         Tensor::Element(vec![1.0, 2.0]),
///     ]))
/// );
/// ```
pub fn expand_tensor(tensor: &Tensor, copy_value: u32) -> Tensor {
    let mut new_tensor_vec = Vec::new();

    for _ in 0..copy_value {
        new_tensor_vec.push((*tensor).clone());
    }

    Tensor::Array(new_tensor_vec)
}


/// Broadcasts the specified Tensor to the target shape. The input Tensor __must__ be broadcastable
/// to the target shape and be the same length as the target shape. If you need to pad a Tensor with
/// ones to make it broadcastable to a desired shape, use [`tensorium::tensor_ops::expand_dims()`].
///
/// # Examples
///
/// ```
/// use tensorium::Tensor;
/// use tensorium::tensor_ops::broadcast_tensor;
/// // A 1x2 Tensor
/// let t = Tensor::Array(Vec::from([
///     Tensor::Element(vec![1.0, 2.0])
/// ]));
/// let target_shape: Vec<usize> = vec![4, 2];
///
/// // Broadcasting will copy our element 4 times to fill the 4x2 shape
/// let target_tensor = Tensor::Array(Vec::from([
///     Tensor::Element(vec![1.0, 2.0]),
///     Tensor::Element(vec![1.0, 2.0]),
///     Tensor::Element(vec![1.0, 2.0]),
///     Tensor::Element(vec![1.0, 2.0])
/// ]));
///
/// assert_eq!(broadcast_tensor(&t, &target_shape), target_tensor);
/// ```
///
/// # Panics
///
/// There are two main cases that could cause this function to panic. The first is if the Tensor's
/// shape's length does not match the target shape. The second is if the Tensor and target shape are
/// not broadcastable.
///
/// ```should_panic
/// use tensorium::Tensor;
/// use tensorium::tensor_ops::broadcast_tensor;
///
/// let t = Tensor::Element(vec![1.0, 2.0]);
/// let target_shape: Vec<usize> = vec![1, 1, 2];
///
/// // Will panic because the target shape is of length 3 and the Tensor's shape is of length 1.
/// let output_tensor = broadcast_tensor(&t, &target_shape);
/// ```
///
/// and
///
/// ```should_panic
/// use tensorium::Tensor;
/// use tensorium::tensor_ops::broadcast_tensor;
///
/// let t = Tensor::Array(Vec::from([
///     Tensor::Element(vec![1.0, 2.0])
/// ]));
/// let target_shape: Vec<usize> = vec![1, 3];
///
/// // Will panic because the target shape is [1, 3] but the Tensor's shape is [1, 2]
/// let output_tensor = broadcast_tensor(&t, &target_shape);
/// ```
///
/// There are two other cases that could cause a panic, but should never occur. They are if a
/// `Tensor::Array` indexes to a `f64` or if a `Tensor::Element` indexes to a `Tensor`. This should
/// never occur but due to the indexing method currently in place, we have to run a `match` call to
/// ensure we have the correct [`tensorium::TensorIndexResult`] type.
pub fn broadcast_tensor(tensor: &Tensor, target_shape: &Vec<usize>) -> Tensor {
    let tensor_shape = get_dimension(&tensor);

    // Check the Tensor's length against the target shape.
    if tensor_shape.len() != target_shape.len() {
        panic!("Tensor shapes are mismatched");
    }

    // Check that they are broadcastable
    if !is_broadcastable(&tensor_shape, &target_shape) {
        panic!("Tensor shapes are not broadcastable");
    }

    // Get the current, left-most dimension
    let current_target_dim = target_shape[0];
    let current_tensor_dim = tensor_shape[0];

    // There are two different paths depending on what Tensor we're working with. If it's an Array,
    // we need to drill deeper until we get an element. If it's an Element, we return the value.
    match tensor {
        Tensor::Array(_) => {
            // Initialize our new Tensor
            let mut new_vec: Vec<Tensor> = Vec::new();

            // For each dimension in this target dim, we need to fill our new Tensor with a value
            for n in 0..current_target_dim {

                // If the current Tensor is of shape 1 at this dimension, copy it by only using
                // the Tensor at index[0]. Else use the Tensor at index[n]. We don't have to worry
                // about indexing out of bounds here and causing a panic because we verified that
                // the dimensions will be either the same or 1 here when we checked that they are
                // broadcastable before the match call.
                let tensor_index = if current_tensor_dim == 1 {0} else {n};
                let next_shape = target_shape.get(1..).unwrap().to_vec();

                // A Tensor::Array will only ever return a Tensor from an index operation
                let tensor_value = match tensor.index(tensor_index) {
                    Some(TensorIndexResult::Tensor(tensor)) => tensor,
                    _ => panic!("Somehow trying to return a float from indexing a Tensor::Array")
                };

                // Because we have a Tensor::Array, we need to drill down to the next level by
                // recursively calling broadcast_tensor on the next shape and next Tensor, while
                // simultaneously pushing the result into our new Vector.
                new_vec.push(
                    broadcast_tensor(&tensor_value, &next_shape)
                );
            }

            Tensor::Array(new_vec)
        },
        Tensor::Element(_) => {
            // Initialize our new Tensor
            let mut new_vec: Vec<f64> = Vec::new();

            // Loop through the target dim.
            for n in 0..current_target_dim {

                // If we only have one element in our Tensor, we will just copy that element.
                // Otherwise, we will use the element at index [n]. Similar to when we work on
                // Tensor::Arrays in the match call, we don't need to worry about an index being
                // out of bounds since we checked that they're broadcastable.
                let tensor_index = if current_tensor_dim == 1 {0} else {n};

                // A Tensor::Element will always index to a TensorIndexResult::Value
                let tensor_value = match tensor.index(tensor_index) {
                    Some(TensorIndexResult::Value(val)) => val,
                    _ => panic!("Somehow trying to return a Tensor from indexing a Tensor::Element")
                };

                new_vec.push(tensor_value);
            }

            Tensor::Element(new_vec)
        }
    }
}

/// Adds a number size 1 dimensions to the front of the shape of the Tensor. This is used to prepare
/// a smaller shaped Tensor for broadcasting to a shape with more dimensions.
///
/// # Examples
///
/// ```
/// use tensorium::Tensor;
/// use tensorium::tensor_ops::{ expand_dims, get_dimension };
///
/// let t1 = Tensor::Element(vec![2.0, 5.0]);
/// let t2 = expand_dims(t1, 2);
/// let shape = get_dimension(&t2);
/// assert_eq!(shape, vec![1, 1, 2]);
/// ```
pub fn expand_dims(tensor: Tensor, num_expansions: usize) -> Tensor {
    let mut tensor = tensor;

    for _ in 0..num_expansions {
        tensor = Tensor::Array(Vec::from([tensor]));
    }

    tensor
}