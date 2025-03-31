
use std::ops::{Add, Sub, Mul, Div, Rem, Range};
use crate::tensor_ops::{
    add_tensors,
    subtract_tensors,
    multiply_tensors,
    divide_tensors,
    remainder_tensors
};

#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Clone)]
pub enum Tensor
{
    Array(Vec<Tensor>),
    Element(Vec<f64>),
}

pub enum TensorIndexResult {
    Tensor(Tensor),
    Value(f64)
}

impl Tensor {
    pub fn index(&self, i: usize) -> Option<TensorIndexResult> {
        match self {
            Tensor::Array(vec) => vec.get(i).cloned().map(TensorIndexResult::Tensor),
            Tensor::Element(vec) => vec.get(i).copied().map(TensorIndexResult::Value)
        }
    }

    pub fn slice(&self, range: Range<usize>) -> Tensor {
        match self {
            Tensor::Array(vec) => Tensor::Array(vec[range].to_vec()),
            Tensor::Element(vec) => Tensor::Element(vec[range].to_vec())
        }
    }

}


impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        add_tensors(&self, &rhs)
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        subtract_tensors(&self, &rhs)
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        multiply_tensors(&self, &rhs)
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        divide_tensors(&self, &rhs)
    }
}

impl Rem for Tensor {
    type Output = Tensor;

    fn rem(self, rhs: Self) -> Self::Output {
        remainder_tensors(&self, &rhs)
    }
}