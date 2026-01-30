import argparse
import os
import logging
import sys
import datetime
from tqdm import tqdm
from model.dncnn import DnCNN

#pytorch stuff
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

""" basic FSDP2 training pipeline completed with data loader and dncnn """

# object to keep track of some of the hyperparam, one object each process
class TrainerParams:
    def __init__(
        self,
        Rank: int,
        Device: torch.device,
        BatchSize: int,
        GradientAccumulationIteration: int,
        SaveEvery: int
    ) -> None:
        self.device = Device
        self.local_rank = Rank
        self.save_every = SaveEvery
        self.batch_size = BatchSize
        self.gradient_accumulation_iter = GradientAccumulationIteration

        logging.info(f'{self.device}, {self.local_rank}, {self.save_every}, {self.batch_size}, {self.gradient_accumulation_iter}')

# house keeping stuff, 
def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """ verification that we have at least 2 gpus to run dist examples """
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus

# init process group as ddp, default backend is nccl on linux with 5080rtx
def fsdp2_setup(device, rank):
    """ init process group using default backend """

    backend = dist.get_default_backend_for_device(device)
    dist.init_process_group(backend=backend, device_id=device, rank=rank)

# prefetch
def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)

def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)

def TrainingProcessMain(args):
    """ This is the main body of the process, should be launched by torchrun """
    
    # enumerate cuda devices
    # setup process group
    # load model as usual
    # fully shard model layer
    # fully shard model
    # assign model to local device allocated to this process, this must happen AFTER fully shard
    # create optimizaer etc and loss function etc
    # set up dataloader with distributed sampler
    # loop epoch
    #   init sampler
    #   loop dataloader data
    #     forward pass 
    #     calc loss
    #     backward pass
    #     optimizer step
    # destroy process group to clean mem

    logging.info(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}')

    # note the number of GPU should be equal or greater than what is specified in torchrun's --nproc_per_node
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        logging.info(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        exit()

    rank = int(os.environ["LOCAL_RANK"])

    if torch.accelerator.is_available():
        my_device_type = torch.accelerator.current_accelerator()    #should be cuda 
        my_device = torch.device(f"{my_device_type}:{rank}")
        torch.accelerator.device_index(rank)
        logging.info(f"Running on rank {rank} on device {my_device}")
    else:
        my_device = torch.device("cpu")
        logging.info(f"Running on device {my_device}")

    # instantiate a local object just to keep tracks of all the params
    local_train_param = TrainerParams(Rank=rank, Device=my_device, BatchSize=args.batch_size, GradientAccumulationIteration=args.gradient_accumulation_iter, SaveEvery=args.save_every)
    
    # Setting things up
    #---------------------

    # init process group using default backend, this is required for DDP and FSDP
    fsdp2_setup(device=local_train_param.device, rank=local_train_param.local_rank)

    # instantiate a model as per normal
    model = DnCNN(channels=3, num_of_layers=17)
    if local_train_param.local_rank == 0:
        logging.info(f'{model}')        

    # this holds param that controls how the model is sharded
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    # decompose model to shards
    for layer in model.dncnn:   #this particular model put layers in its attribute .dncnn
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    
    # create the optimizer only afte model is fully_shard
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    L1_loss = torch.nn.L1Loss().to(local_train_param.device)

    # only assign model to local cuda device after fully shard
    model.to(local_train_param.device) 

    # prefetch settings
    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)
    
    # data loader stuff, as usual except there is a DistributedSampler is used in dataloader.
    #--------------------------
    # getting data from disk to the dataset, 
    # note the dataset need to be initialized before the distributed sampler and dataloader. 
    # how to intialize dataset is not important, just as usual
    filename = '/media/fred/DATA_SSD/denoise_datasets/s256_clean_noisy_dataset1.ds'
    dataset = torch.load(filename, weights_only=False)

    dSampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=local_train_param.local_rank,
        shuffle=True, # Recommended for FSDP
        seed=42
    )
    
    # data loader's shuffle is done by the distributed sampler
    trainning_dataloader = DataLoader(dataset= dataset, batch_size=local_train_param.batch_size, shuffle=False, sampler=dSampler, num_workers=2, pin_memory=True, drop_last=False) #num_worker forced more paralleism between learning and data loading (gpu v cpu tasks). seems 1 is ok, more than 1 causes lot of memory usage (160GB)
    
    # training loop
    #--------------------------
    # doing epoch stuff
    for epoch in range(10):
        timenow = datetime.datetime.now()
        trainning_dataloader.sampler.set_epoch(epoch) #init sampler for each epoch

        # only need to tqdm one process, processes are synchronized by forward and backward pass
        tqdm_dataloader = trainning_dataloader
        if local_train_param.local_rank == 0:
            tqdm_dataloader = tqdm(trainning_dataloader)
        
        #loop data inside dataloader to train
        for i, data in enumerate(tqdm_dataloader, 0):
            
            #send input and expected output to the GPU
            source = data[0].to(local_train_param.device) 
            targets = data[1].to(local_train_param.device) 

            if args.explicit_prefetching:
                model.unshard()
            
            #foward pass, works out the output 
            output = model(source)

            #works out the loss between the actual model output and the expected output
            loss = L1_loss(output, targets)

            #torch handles all the sync suff by itself 
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad()

        deltatime = datetime.datetime.now() - timenow
        logging.info(f'epoch {epoch} rank {local_train_param.local_rank}, epoch time {deltatime.total_seconds()}')

    # clean up the process group before exiting
    dist.destroy_process_group()


if __name__ == "__main__":
    
    """ torchrun --nproc_per_node 2 ./fsdp2_basic.py --save-every=20 --batch-size=10 """

    parser = argparse.ArgumentParser(description="PyTorch FSDP2 trainer params")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False) #if prefetching is specified, a lot more memory is used, which causes the cuda out of memory unless tune down the batch size
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--gradient-accumulation-iter", type=int, default=0)
    args = parser.parse_args()

    # setup logging to file and stdout
    logging.basicConfig(filename='myapp.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) #log to stdout and a file
    
    TrainingProcessMain(args)