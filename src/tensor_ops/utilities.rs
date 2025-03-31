use crate::Tensor;

/// Finds the shape of the input Tensor.
///
/// # Examples
///
/// ```
/// use tensorium::Tensor;
/// use tensorium::tensor_ops::get_dimension;
///
/// let t1 = Tensor::Array(Vec::from([
///     Tensor::Element(vec![1.0, 2.0, 3.0]),
///     Tensor::Element(vec![4.0, 5.0, 6.0])
/// ]));
/// let shape = get_dimension(&t1);
/// assert_eq!(shape, Vec::from([2, 3]));
/// ```
pub fn get_dimension(tensor: &Tensor) -> Vec<usize>{
    let mut dimensions = Vec::new();

    match tensor {
        Tensor::Element(x) => dimensions.push(x.len()),
        Tensor::Array(x) => {
            let subdims: Vec<Vec<usize>> = x.iter()
                .map(|t| get_dimension(&t))
                .collect();
            if !subdims.first().map(|first| subdims.iter().all(|y| y == first)).unwrap_or(true){
                panic!("Dimensions do not match!")
            }

            let mut dims = Vec::from([subdims.len()]);
            for subdim in subdims[0].iter() {
                dims.push(*subdim)
            }
            dimensions = dims;
        }
    };

    dimensions
}