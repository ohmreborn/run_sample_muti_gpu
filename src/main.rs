pub mod all_reduce;


use anyhow::{bail, Result};
use clap::Parser;

use candle_nn::{linear, Linear, VarBuilder, VarMap, Module, loss, SGD, Optimizer};
use candle_core::{DType, Device, Tensor, Var, Shape};

use crate::all_reduce::*;

use cudarc::driver::safe::CudaDevice;
use cudarc::nccl::safe::{Comm, Id};
use std::io::Write;
use std::rc::Rc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    num_shards: usize,

    #[arg(long)]
    rank: Option<usize>,

    #[arg(long, default_value = "nccl_id.txt")]
    comm_file: String,
}

fn get_batches(x: Tensor, y: Tensor, batch_size: usize) -> Result<Vec<(Tensor, Tensor)>> {
    let n_samples = x.dims()[0];
    let mut batches: Vec<(Tensor, Tensor)> = Vec::new();

    for i in (0..n_samples).step_by(batch_size) {
        let end = (i + batch_size).min(n_samples);
        let x_batch = x.narrow(0, i, end - i)?;
        let y_batch = y.narrow(0, i, end - i)?;
        batches.push((x_batch, y_batch));
    }

    Ok(batches)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let _dtype = DType::F32;

	let comm_file = std::path::PathBuf::from(&args.comm_file);
	if comm_file.exists() {
		bail!("comm file {comm_file:?} already exists, please remove it first")
	}

    let rank = match args.rank {
        None => {
            println!("creating {} child processes", args.num_shards);
            let children: Vec<_> = (0..args.num_shards)
                .map(|rank| {
                    let mut args: std::collections::VecDeque<_> = std::env::args().collect();
                    args.push_back("--rank".to_string());
                    args.push_back(format!("{rank}"));
                    let name = args.pop_front().unwrap();
                    std::process::Command::new(name).args(args).spawn().unwrap()
                })
            .collect();
            for mut child in children {
                child.wait()?;
            }
            return Ok(());
        }
        Some(rank) => rank,
    };

    let num_shards = args.num_shards;
    // Primitive IPC
    let id = if rank == 0 {
        let id = Id::new().unwrap();
        let tmp_file = comm_file.with_extension(".comm.tgz");
        std::fs::File::create(&tmp_file)?
            .write_all(&id.internal().iter().map(|&i| i as u8).collect::<Vec<_>>())?;
        std::fs::rename(&tmp_file, &comm_file)?;
        id
    } else {
        while !comm_file.exists() {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        let data = std::fs::read(&comm_file)?;
        let internal: [i8; 128] = data
            .into_iter()
            .map(|i| i as i8)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let id: Id = Id::uninit(internal);
        id
    };

    let device = CudaDevice::new(rank)?;
    let comm = match Comm::from_rank(device, rank, num_shards, id) {
        Ok(comm) => Rc::new(comm),
        Err(err) => anyhow::bail!("nccl error {:?}", err.0),
    };
    if rank == 0 {
        std::fs::remove_file(comm_file)?;
    }
    println!("Rank {rank:?} spawned");
    let device = Device::new_cuda(rank)?;
    let all_reducee = AllReduce{ comm};

//     let a = Var::ones(3, DType::F32, &device)?;
//     let b = a.apply_op1_no_bwd(&all_reducee)?;
//     println!("{}",b);

    let batch_size:usize = 20;

    // create a data
    let n_sample:f32 = 100.0;
    let div: Tensor = Tensor::new(&[n_sample],&device)?;
    let x: Tensor = Tensor::arange(0f32, n_sample, &device)?
        .reshape((100,1))?
        .broadcast_div(&div)?;
    let m: Tensor = Tensor::new(&[[3f32]], &device)?;
    let c: Tensor = Tensor::new(&[[3f32]], &device)?;
    let noise: Tensor = Tensor::rand(0f32, 1., Shape::from((100, 1)), &device)?;
    let y: Tensor = x.broadcast_mul(&m)?
        .broadcast_add(&c)?
        .broadcast_add(&noise)?;
    let data_loader: Vec<(Tensor,Tensor)> = get_batches(x, y, batch_size)?;

    // initialize a model optimizer
    let varmap: VarMap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model: Linear = linear(1, 1, vb)?;
    let mut vars:Vec<Var> = varmap.all_vars();
	vars.sort_by_key(|v| v.id().get_id());
	let mut optimizer: SGD = SGD::new(vars.clone(), 0.01)?;

	for var in vars.iter(){
		println!("{} {}",var, var.is_contiguous());
	}

	for var in vars.iter(){
		let v = var.contiguous()?.apply_op1_no_bwd(&all_reducee)?;
		var.set(&v)?;
	}

	for var in vars.iter(){
		println!("{} {}",var, var.is_contiguous());
	}

	// training a model
	for _epoch in 0..10 {
		for (x_train,y_train) in &data_loader{
			let pred: Tensor = model.forward(&x_train)?;
			let loss_res: Tensor = loss::mse(&pred,&y_train)?;
			optimizer.backward_step(&loss_res)?;

			for var in vars.iter(){
				let v = var.contiguous()?.apply_op1_no_bwd(&all_reducee)?;
				var.set(&v)?;
			}

			println!("{}",loss_res);
		}
	}

    Ok(())
}

