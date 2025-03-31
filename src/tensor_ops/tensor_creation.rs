use crate::Tensor;

/// # DO NOT USE IDK WHY THIS IS HERE I DON'T REMEMBER CREATING IT
pub fn zero_tensor(shape: Vec<usize>) -> Tensor {
    if shape.len() == 1 {
        let mut outvec: Vec<f64> = Vec::new();
        for _ in 0..shape[0] {
            outvec.push(0.0);
        }
        Tensor::Element(outvec)
    } else {
        let mut outvec: Vec<Tensor> = Vec::new();
        for _ in 0..shape[0] {
            let next_shape = Vec::from(&shape[1..]);
            outvec.push(zero_tensor(next_shape));
        }
        Tensor::Array(outvec)
    }
}