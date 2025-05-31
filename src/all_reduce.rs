use candle_core::{CpuStorage, CustomOp1, DType, Layout, Shape, Result};
use cudarc::nccl::safe::{Comm, ReduceOp};
use candle_core::backend::BackendStorage;

use std::rc::Rc;

pub struct AllReduce {
    pub comm: Rc<Comm>,
}

/// This is actually not safe: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
/// But for this example purposes, this will work
unsafe impl Sync for AllReduce {}
unsafe impl Send for AllReduce {}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "allreduce"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllReduce is never used on cpu")
    }

    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::WrapErr;
        use cudarc::driver::DeviceSlice;

		let elem_count = l.shape().elem_count();
		let dev = s.device().clone();
		let dst = match s.dtype() {
			DType::F32 => {
				let s = s.as_cuda_slice::<f32>()?;

				let mut dst = unsafe { dev.alloc::<f32>(elem_count) }.w()?;
				self.comm
					.all_reduce(s, &mut dst, &ReduceOp::Avg)
					.map_err(candle_core::Error::debug)?;
				candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
			}
			dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
		};
		Ok((dst, l.shape().clone()))
	}
}
