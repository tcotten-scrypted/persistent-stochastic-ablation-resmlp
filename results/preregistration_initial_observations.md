## 1*2048 Comprehensive Results

###Results Shape {1*2048} Parameters {1628170}
* None: Mean=98.12% | Std=0.07% | Min=98.01% | Max=98.28% (n=10)
* Decay: Mean=98.14% | Std=0.05% | Min=98.08% | Max=98.23% (n=10)
* Dropout: Mean=98.22% | Std=0.06% | Min=98.14% | Max=98.30% (n=10)
* Full: Mean=98.13% | Std=0.06% | Min=98.03% | Max=98.25% (n=10)
* Hidden: Mean=98.19% | Std=0.09% | Min=98.03% | Max=98.33% (n=10)
* Output: Mean=98.15% | Std=0.07% | Min=98.08% | Max=98.30% (n=10)

### Architecture: 1*2048

| Trial | None | Decay | Dropout | Full | Hidden | Output |
|:-----::----:|:----:|:----:|:----:|:----:|:----:|
| 1 | 98.07% | 98.08% | 98.25% | 98.17% | 98.15% | 98.14% |
| 2 | 98.04% | 98.11% | 98.30% | 98.08% | 98.03% | 98.12% |
| 3 | 98.09% | 98.16% | 98.19% | 98.10% | 98.12% | 98.30% |
| 4 | 98.18% | 98.20% | 98.17% | 98.13% | 98.22% | 98.13% |
| 5 | 98.13% | 98.14% | 98.28% | 98.08% | 98.29% | 98.28% |
| 6 | 98.16% | 98.10% | 98.27% | 98.25% | 98.21% | 98.10% |
| 7 | 98.01% | 98.09% | 98.16% | 98.18% | 98.24% | 98.08% |
| 8 | 98.12% | 98.14% | 98.14% | 98.10% | 98.08% | 98.09% |
| 9 | 98.16% | 98.14% | 98.26% | 98.15% | 98.33% | 98.13% |
| 10 | 98.28% | 98.23% | 98.15% | 98.03% | 98.22% | 98.15% |

## 32*4 Manual Tests (Single Trials of All Modes)

None: 27.7% @ 2
Decay: 49.08% @ 56
Dropout: 29.87% @ 2 (11.47% @ 1 on a stuck run)
Full: 82.56% @ 100 (first run was stuck for a bit on 9.84%, second run improvements to the very end!)
Hidden: 64.49% @ 90
Output: 45.51% @ 35

Example training in PSA on 32*4
(src-py3.11) (base) timcotten@Tims-Laptop-2 002-persistent-stochastic-ablation-resmlp % poetry run clean; poetry run train --arch="[32*4]" --ablation-mode full
Warning: 'clean' is an entry point defined in pyproject.toml, but it's not installed as a script. You may get improper `sys.argv[0]`.

The support to run uninstalled scripts will be removed in a future release.

Run `poetry install` to resolve and get rid of this message.

                    Files to Remove                     
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ File Path                    ┃         Size ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━┩
│ models/mnist_lkg.safetensors │ 69,259 bytes │ Exists │
└──────────────────────────────┴──────────────┴────────┘
Total size to remove: 69,259 bytes
╭──────────────────────────────────────────────────────────────────────────── Clean Complete ─────────────────────────────────────────────────────────────────────────────╮
│ ✅ Successfully removed 1 file(s)                                                                                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Warning: 'train' is an entry point defined in pyproject.toml, but it's not installed as a script. You may get improper `sys.argv[0]`.

The support to run uninstalled scripts will be removed in a future release.

Run `poetry install` to resolve and get rid of this message.

[20:06:31] INFO     Ablator (full mode) indexed 66 total linear layers.                                                                                                    
╭─────────────────────────────────────────╮
│ Frustration Engine: MNIST Ablation Test │
╰─── Ablation Mode: full | Device: mps ───╯
                        ResMLP Architecture Summary: [32*4]                        
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━┳━━━━━━━┳━━━━━━━━┓
┃   Metric   ┃ Initial Proj. ┃ ResStack (32x4) ┃ Final Proj. ┃   ┃       ┃        ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━╇━━━━━━━╇━━━━━━━━┩
│   Shape    │      784      │     784 → 4     │      ✓      │ ✓ │ 4 → 4 │ 4 → 10 │
│ Parameters │       -       │      3,140      │      -      │ - │ 1,280 │ 50     │
└────────────┴───────────────┴─────────────────┴─────────────┴───┴───────┴────────┘
[20:06:35] INFO     Model has 4,470 parameters.                                                                                                                            
           INFO     Training for 100 meta-loops.                                                                                                                           
           INFO     📊 Dataset splits: 50k train, 10k validation, 10k test                                                                                                 
           INFO     🎯 Meta-loops use validation accuracy for LKG decisions                                                                                                
           INFO     🧪 Final test accuracy reported at completion                                                                                                          
           WARNING  No checkpoint found at models/mnist_lkg.safetensors. Starting from scratch.                                                                            
╭────────────────────────────────────────────────────────────────────── Meta-Loop 1/100 (Global: 1) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    -1.00%                                                                                                                            │
│   Current Loop Validation Accuracy    16.93%                                                                                                                            │
│   Global Meta-Loop                    1                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:06:41] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 16.93% @ 1                                                                                                   
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.8.layers.0'.                                                                 

╭────────────────────────────────────────────────────────────────────── Meta-Loop 2/100 (Global: 2) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    16.93%                                                                                                                            │
│   Current Loop Validation Accuracy    26.14%                                                                                                                            │
│   Global Meta-Loop                    2                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:06:47] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 26.14% @ 2                                                                                                   
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.14.layers.0'.                                                                

╭────────────────────────────────────────────────────────────────────── Meta-Loop 3/100 (Global: 3) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    26.14%                                                                                                                            │
│   Current Loop Validation Accuracy    29.15%                                                                                                                            │
│   Global Meta-Loop                    3                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:06:53] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 29.15% @ 3                                                                                                   
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.12.layers.0'.                                                                

╭────────────────────────────────────────────────────────────────────── Meta-Loop 4/100 (Global: 4) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    29.15%                                                                                                                            │
│   Current Loop Validation Accuracy    35.97%                                                                                                                            │
│   Global Meta-Loop                    4                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:06:59] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 35.97% @ 4                                                                                                   
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
[20:07:00] INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.13.layers.0'.                                                                

╭────────────────────────────────────────────────────────────────────── Meta-Loop 5/100 (Global: 5) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    35.97%                                                                                                                            │
│   Current Loop Validation Accuracy    38.10%                                                                                                                            │
│   Global Meta-Loop                    5                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:07:06] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 38.10% @ 5                                                                                                   
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.8.layers.0'.                                                                 

╭────────────────────────────────────────────────────────────────────── Meta-Loop 6/100 (Global: 6) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    38.10%                                                                                                                            │
│   Current Loop Validation Accuracy    39.60%                                                                                                                            │
│   Global Meta-Loop                    6                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:07:13] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 39.60% @ 6                                                                                                   
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.8.layers.3'.                                                                 

╭────────────────────────────────────────────────────────────────────── Meta-Loop 7/100 (Global: 7) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    39.60%                                                                                                                            │
│   Current Loop Validation Accuracy    39.93%                                                                                                                            │
│   Global Meta-Loop                    7                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:07:19] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 39.93% @ 7                                                                                                   
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.10.layers.0'.                                                                

╭────────────────────────────────────────────────────────────────────── Meta-Loop 8/100 (Global: 8) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    39.93%                                                                                                                            │
│   Current Loop Validation Accuracy    40.07%                                                                                                                            │
│   Global Meta-Loop                    8                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:07:25] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 40.07% @ 8                                                                                                   
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.21.layers.3'.                                                                

╭────────────────────────────────────────────────────────────────────── Meta-Loop 9/100 (Global: 9) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    40.07%                                                                                                                            │
│   Current Loop Validation Accuracy    40.19%                                                                                                                            │
│   Global Meta-Loop                    9                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:07:31] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 40.19% @ 9                                                                                                   
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.1.layers.0'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 10/100 (Global: 10) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    40.19%                                                                                                                            │
│   Current Loop Validation Accuracy    40.48%                                                                                                                            │
│   Global Meta-Loop                    10                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:07:38] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 40.48% @ 10                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.0'.                                                                                   

╭───────────────────────────────────────────────────────────────────── Meta-Loop 11/100 (Global: 11) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    40.48%                                                                                                                            │
│   Current Loop Validation Accuracy    40.35%                                                                                                                            │
│   Global Meta-Loop                    11                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[20:07:44] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.13.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 12/100 (Global: 12) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    40.48%                                                                                                                            │
│   Current Loop Validation Accuracy    40.76%                                                                                                                            │
│   Global Meta-Loop                    12                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:07:50] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 40.76% @ 12                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.3.layers.0'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 13/100 (Global: 13) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    40.76%                                                                                                                            │
│   Current Loop Validation Accuracy    40.77%                                                                                                                            │
│   Global Meta-Loop                    13                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:07:56] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 40.77% @ 13                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.26.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 14/100 (Global: 14) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    40.77%                                                                                                                            │
│   Current Loop Validation Accuracy    41.37%                                                                                                                            │
│   Global Meta-Loop                    14                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:08:02] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 41.37% @ 14                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.17.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 15/100 (Global: 15) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    41.37%                                                                                                                            │
│   Current Loop Validation Accuracy    41.72%                                                                                                                            │
│   Global Meta-Loop                    15                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:08:08] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 41.72% @ 15                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.30.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 16/100 (Global: 16) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    41.72%                                                                                                                            │
│   Current Loop Validation Accuracy    42.44%                                                                                                                            │
│   Global Meta-Loop                    16                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:08:15] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 42.44% @ 16                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.25.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 17/100 (Global: 17) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    42.44%                                                                                                                            │
│   Current Loop Validation Accuracy    44.52%                                                                                                                            │
│   Global Meta-Loop                    17                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:08:21] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 44.52% @ 17                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.13.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 18/100 (Global: 18) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    44.52%                                                                                                                            │
│   Current Loop Validation Accuracy    47.07%                                                                                                                            │
│   Global Meta-Loop                    18                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:08:27] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 47.07% @ 18                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.29.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 19/100 (Global: 19) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    47.07%                                                                                                                            │
│   Current Loop Validation Accuracy    47.84%                                                                                                                            │
│   Global Meta-Loop                    19                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:08:33] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 47.84% @ 19                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.21.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 20/100 (Global: 20) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    47.84%                                                                                                                            │
│   Current Loop Validation Accuracy    48.60%                                                                                                                            │
│   Global Meta-Loop                    20                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:08:39] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 48.60% @ 20                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.7.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 21/100 (Global: 21) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    48.60%                                                                                                                            │
│   Current Loop Validation Accuracy    49.54%                                                                                                                            │
│   Global Meta-Loop                    21                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:08:45] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 49.54% @ 21                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.5.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 22/100 (Global: 22) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    49.54%                                                                                                                            │
│   Current Loop Validation Accuracy    49.84%                                                                                                                            │
│   Global Meta-Loop                    22                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:08:51] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 49.84% @ 22                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.6.layers.0'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 23/100 (Global: 23) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    49.84%                                                                                                                            │
│   Current Loop Validation Accuracy    50.95%                                                                                                                            │
│   Global Meta-Loop                    23                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:08:57] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 50.95% @ 23                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.5.layers.0'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 24/100 (Global: 24) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    50.95%                                                                                                                            │
│   Current Loop Validation Accuracy    51.13%                                                                                                                            │
│   Global Meta-Loop                    24                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:09:03] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 51.13% @ 24                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.16.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 25/100 (Global: 25) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    51.13%                                                                                                                            │
│   Current Loop Validation Accuracy    51.48%                                                                                                                            │
│   Global Meta-Loop                    25                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:09:09] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 51.48% @ 25                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.25.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 26/100 (Global: 26) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    51.48%                                                                                                                            │
│   Current Loop Validation Accuracy    52.17%                                                                                                                            │
│   Global Meta-Loop                    26                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:09:15] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 52.17% @ 26                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.21.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 27/100 (Global: 27) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    52.17%                                                                                                                            │
│   Current Loop Validation Accuracy    52.41%                                                                                                                            │
│   Global Meta-Loop                    27                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:09:21] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 52.41% @ 27                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.27.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 28/100 (Global: 28) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    52.41%                                                                                                                            │
│   Current Loop Validation Accuracy    53.63%                                                                                                                            │
│   Global Meta-Loop                    28                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:24:28] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 53.63% @ 28                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.2.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 29/100 (Global: 29) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    53.63%                                                                                                                            │
│   Current Loop Validation Accuracy    55.15%                                                                                                                            │
│   Global Meta-Loop                    29                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[20:41:06] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 55.15% @ 29                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.27.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 30/100 (Global: 30) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    55.15%                                                                                                                            │
│   Current Loop Validation Accuracy    55.24%                                                                                                                            │
│   Global Meta-Loop                    30                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[21:13:00] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 55.24% @ 30                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.31.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 31/100 (Global: 31) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    55.24%                                                                                                                            │
│   Current Loop Validation Accuracy    56.92%                                                                                                                            │
│   Global Meta-Loop                    31                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[21:29:56] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 56.92% @ 31                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.25.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 32/100 (Global: 32) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    56.92%                                                                                                                            │
│   Current Loop Validation Accuracy    57.36%                                                                                                                            │
│   Global Meta-Loop                    32                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[21:47:03] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 57.36% @ 32                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.10.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 33/100 (Global: 33) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    57.36%                                                                                                                            │
│   Current Loop Validation Accuracy    57.91%                                                                                                                            │
│   Global Meta-Loop                    33                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[21:50:02] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 57.91% @ 33                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.1.layers.0'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 34/100 (Global: 34) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    57.91%                                                                                                                            │
│   Current Loop Validation Accuracy    59.66%                                                                                                                            │
│   Global Meta-Loop                    34                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:01:03] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 59.66% @ 34                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.27.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 35/100 (Global: 35) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    59.66%                                                                                                                            │
│   Current Loop Validation Accuracy    60.97%                                                                                                                            │
│   Global Meta-Loop                    35                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:01:14] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 60.97% @ 35                                                                                                  
[22:01:15] INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.12.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 36/100 (Global: 36) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    60.97%                                                                                                                            │
│   Current Loop Validation Accuracy    62.56%                                                                                                                            │
│   Global Meta-Loop                    36                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:01:26] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 62.56% @ 36                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.14.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 37/100 (Global: 37) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.13%                                                                                                                            │
│   Global Meta-Loop                    37                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:04:07] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.12.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 38/100 (Global: 38) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.26%                                                                                                                            │
│   Global Meta-Loop                    38                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:04:13] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 7 in layer 'layer_stack.4'.                                                                                   

╭───────────────────────────────────────────────────────────────────── Meta-Loop 39/100 (Global: 39) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    58.24%                                                                                                                            │
│   Global Meta-Loop                    39                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:04:19] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.0.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 40/100 (Global: 40) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.66%                                                                                                                            │
│   Global Meta-Loop                    40                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:04:25] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.21.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 41/100 (Global: 41) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.52%                                                                                                                            │
│   Global Meta-Loop                    41                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:04:31] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.27.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 42/100 (Global: 42) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.64%                                                                                                                            │
│   Global Meta-Loop                    42                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:04:37] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.9.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 43/100 (Global: 43) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.30%                                                                                                                            │
│   Global Meta-Loop                    43                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:04:42] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.20.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 44/100 (Global: 44) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.94%                                                                                                                            │
│   Global Meta-Loop                    44                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:04:48] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.5.layers.0'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 45/100 (Global: 45) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.97%                                                                                                                            │
│   Global Meta-Loop                    45                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:04:54] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.11.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 46/100 (Global: 46) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.02%                                                                                                                            │
│   Global Meta-Loop                    46                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:05:00] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.17.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 47/100 (Global: 47) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.85%                                                                                                                            │
│   Global Meta-Loop                    47                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:05:06] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.28.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 48/100 (Global: 48) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.92%                                                                                                                            │
│   Global Meta-Loop                    48                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:05:12] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.22.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 49/100 (Global: 49) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.44%                                                                                                                            │
│   Global Meta-Loop                    49                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:05:18] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.6.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 50/100 (Global: 50) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.14%                                                                                                                            │
│   Global Meta-Loop                    50                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:05:24] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.20.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 51/100 (Global: 51) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.45%                                                                                                                            │
│   Global Meta-Loop                    51                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:05:30] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.17.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 52/100 (Global: 52) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.34%                                                                                                                            │
│   Global Meta-Loop                    52                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:05:36] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.21.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 53/100 (Global: 53) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.14%                                                                                                                            │
│   Global Meta-Loop                    53                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:05:41] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.18.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 54/100 (Global: 54) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.33%                                                                                                                            │
│   Global Meta-Loop                    54                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:05:47] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.9.layers.0'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 55/100 (Global: 55) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.99%                                                                                                                            │
│   Global Meta-Loop                    55                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:05:53] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.5.layers.0'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 56/100 (Global: 56) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.80%                                                                                                                            │
│   Global Meta-Loop                    56                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:05:59] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.30.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 57/100 (Global: 57) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.82%                                                                                                                            │
│   Global Meta-Loop                    57                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:06:05] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.29.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 58/100 (Global: 58) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.93%                                                                                                                            │
│   Global Meta-Loop                    58                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:06:11] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.27.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 59/100 (Global: 59) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.94%                                                                                                                            │
│   Global Meta-Loop                    59                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:06:17] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.15.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 60/100 (Global: 60) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.83%                                                                                                                            │
│   Global Meta-Loop                    60                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:06:23] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.1.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 61/100 (Global: 61) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    60.48%                                                                                                                            │
│   Global Meta-Loop                    61                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:06:29] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.3.layers.0'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 62/100 (Global: 62) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.30%                                                                                                                            │
│   Global Meta-Loop                    62                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:06:34] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.19.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 63/100 (Global: 63) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    61.88%                                                                                                                            │
│   Global Meta-Loop                    63                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:06:40] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.6.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 64/100 (Global: 64) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.39%                                                                                                                            │
│   Global Meta-Loop                    64                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:06:46] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.10.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 65/100 (Global: 65) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.02%                                                                                                                            │
│   Global Meta-Loop                    65                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:06:52] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.0.layers.0'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 66/100 (Global: 66) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.56%                                                                                                                            │
│   Current Loop Validation Accuracy    62.59%                                                                                                                            │
│   Global Meta-Loop                    66                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:06:58] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 62.59% @ 66                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.12.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 67/100 (Global: 67) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.59%                                                                                                                            │
│   Current Loop Validation Accuracy    62.99%                                                                                                                            │
│   Global Meta-Loop                    67                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:07:04] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 62.99% @ 67                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.12.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 68/100 (Global: 68) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    62.99%                                                                                                                            │
│   Current Loop Validation Accuracy    63.52%                                                                                                                            │
│   Global Meta-Loop                    68                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:07:10] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 63.52% @ 68                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.13.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 69/100 (Global: 69) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    63.52%                                                                                                                            │
│   Current Loop Validation Accuracy    64.36%                                                                                                                            │
│   Global Meta-Loop                    69                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:07:17] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 64.36% @ 69                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.22.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 70/100 (Global: 70) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    64.36%                                                                                                                            │
│   Current Loop Validation Accuracy    64.62%                                                                                                                            │
│   Global Meta-Loop                    70                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:07:23] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 64.62% @ 70                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.7.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 71/100 (Global: 71) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    64.62%                                                                                                                            │
│   Current Loop Validation Accuracy    64.67%                                                                                                                            │
│   Global Meta-Loop                    71                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:07:29] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 64.67% @ 71                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.29.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 72/100 (Global: 72) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    64.67%                                                                                                                            │
│   Current Loop Validation Accuracy    65.22%                                                                                                                            │
│   Global Meta-Loop                    72                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:07:35] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 65.22% @ 72                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.0.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 73/100 (Global: 73) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    65.22%                                                                                                                            │
│   Current Loop Validation Accuracy    66.43%                                                                                                                            │
│   Global Meta-Loop                    73                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:07:41] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 66.43% @ 73                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.14.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 74/100 (Global: 74) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    66.43%                                                                                                                            │
│   Current Loop Validation Accuracy    65.67%                                                                                                                            │
│   Global Meta-Loop                    74                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:07:47] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.6.layers.0'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 75/100 (Global: 75) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    66.43%                                                                                                                            │
│   Current Loop Validation Accuracy    65.90%                                                                                                                            │
│   Global Meta-Loop                    75                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:07:53] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.13.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 76/100 (Global: 76) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    66.43%                                                                                                                            │
│   Current Loop Validation Accuracy    66.44%                                                                                                                            │
│   Global Meta-Loop                    76                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:08:00] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 66.44% @ 76                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.27.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 77/100 (Global: 77) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    66.44%                                                                                                                            │
│   Current Loop Validation Accuracy    66.32%                                                                                                                            │
│   Global Meta-Loop                    77                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:08:06] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.9.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 78/100 (Global: 78) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    66.44%                                                                                                                            │
│   Current Loop Validation Accuracy    66.69%                                                                                                                            │
│   Global Meta-Loop                    78                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:08:12] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 66.69% @ 78                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.25.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 79/100 (Global: 79) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    66.69%                                                                                                                            │
│   Current Loop Validation Accuracy    67.13%                                                                                                                            │
│   Global Meta-Loop                    79                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:08:18] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 67.13% @ 79                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.29.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 80/100 (Global: 80) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    67.13%                                                                                                                            │
│   Current Loop Validation Accuracy    67.29%                                                                                                                            │
│   Global Meta-Loop                    80                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:08:24] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 67.29% @ 80                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.24.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 81/100 (Global: 81) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    67.29%                                                                                                                            │
│   Current Loop Validation Accuracy    69.82%                                                                                                                            │
│   Global Meta-Loop                    81                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:08:30] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 69.82% @ 81                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.24.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 82/100 (Global: 82) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    69.82%                                                                                                                            │
│   Current Loop Validation Accuracy    70.81%                                                                                                                            │
│   Global Meta-Loop                    82                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:08:36] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 70.81% @ 82                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.0'.                                                                                   

╭───────────────────────────────────────────────────────────────────── Meta-Loop 83/100 (Global: 83) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    70.81%                                                                                                                            │
│   Current Loop Validation Accuracy    56.61%                                                                                                                            │
│   Global Meta-Loop                    83                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:08:42] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 9 in layer 'layer_stack.4'.                                                                                   

╭───────────────────────────────────────────────────────────────────── Meta-Loop 84/100 (Global: 84) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    70.81%                                                                                                                            │
│   Current Loop Validation Accuracy    69.36%                                                                                                                            │
│   Global Meta-Loop                    84                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[22:08:49] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.30.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 85/100 (Global: 85) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    70.81%                                                                                                                            │
│   Current Loop Validation Accuracy    72.13%                                                                                                                            │
│   Global Meta-Loop                    85                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:08:55] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 72.13% @ 85                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.13.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 86/100 (Global: 86) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    72.13%                                                                                                                            │
│   Current Loop Validation Accuracy    73.61%                                                                                                                            │
│   Global Meta-Loop                    86                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:09:01] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 73.61% @ 86                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.12.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 87/100 (Global: 87) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    73.61%                                                                                                                            │
│   Current Loop Validation Accuracy    75.77%                                                                                                                            │
│   Global Meta-Loop                    87                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:09:07] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 75.77% @ 87                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.30.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 88/100 (Global: 88) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    75.77%                                                                                                                            │
│   Current Loop Validation Accuracy    76.68%                                                                                                                            │
│   Global Meta-Loop                    88                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:09:13] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 76.68% @ 88                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.24.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 89/100 (Global: 89) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    76.68%                                                                                                                            │
│   Current Loop Validation Accuracy    77.55%                                                                                                                            │
│   Global Meta-Loop                    89                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:09:20] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 77.55% @ 89                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.2.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 90/100 (Global: 90) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    77.55%                                                                                                                            │
│   Current Loop Validation Accuracy    77.68%                                                                                                                            │
│   Global Meta-Loop                    90                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:09:26] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 77.68% @ 90                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.27.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 91/100 (Global: 91) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    77.68%                                                                                                                            │
│   Current Loop Validation Accuracy    78.64%                                                                                                                            │
│   Global Meta-Loop                    91                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:09:32] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 78.64% @ 91                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.14.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 92/100 (Global: 92) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    78.64%                                                                                                                            │
│   Current Loop Validation Accuracy    78.89%                                                                                                                            │
│   Global Meta-Loop                    92                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:09:38] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 78.89% @ 92                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.18.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 93/100 (Global: 93) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    78.89%                                                                                                                            │
│   Current Loop Validation Accuracy    79.45%                                                                                                                            │
│   Global Meta-Loop                    93                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:09:44] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 79.45% @ 93                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.1.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 94/100 (Global: 94) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    79.45%                                                                                                                            │
│   Current Loop Validation Accuracy    79.46%                                                                                                                            │
│   Global Meta-Loop                    94                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:09:50] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 79.46% @ 94                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.5.layers.0'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 95/100 (Global: 95) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    79.46%                                                                                                                            │
│   Current Loop Validation Accuracy    79.88%                                                                                                                            │
│   Global Meta-Loop                    95                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:09:56] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 79.88% @ 95                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.24.layers.0'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 96/100 (Global: 96) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    79.88%                                                                                                                            │
│   Current Loop Validation Accuracy    80.19%                                                                                                                            │
│   Global Meta-Loop                    96                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:10:03] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 80.19% @ 96                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 1 in layer 'layer_stack.3.blocks.1.layers.3'.                                                                 

╭───────────────────────────────────────────────────────────────────── Meta-Loop 97/100 (Global: 97) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    80.19%                                                                                                                            │
│   Current Loop Validation Accuracy    80.79%                                                                                                                            │
│   Global Meta-Loop                    97                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:10:09] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 80.79% @ 97                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 2 in layer 'layer_stack.3.blocks.21.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 98/100 (Global: 98) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    80.79%                                                                                                                            │
│   Current Loop Validation Accuracy    80.88%                                                                                                                            │
│   Global Meta-Loop                    98                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:10:14] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 80.88% @ 98                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.30.layers.3'.                                                                

╭───────────────────────────────────────────────────────────────────── Meta-Loop 99/100 (Global: 99) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    80.88%                                                                                                                            │
│   Current Loop Validation Accuracy    81.24%                                                                                                                            │
│   Global Meta-Loop                    99                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:10:20] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 81.24% @ 99                                                                                                  
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
[22:10:21] INFO     🧠 (Full Mode) Partially ablating neuron 0 in layer 'layer_stack.3.blocks.18.layers.3'.                                                                

╭──────────────────────────────────────────────────────────────────── Meta-Loop 100/100 (Global: 100) ────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    81.24%                                                                                                                            │
│   Current Loop Validation Accuracy    81.82%                                                                                                                            │
│   Global Meta-Loop                    100                                                                                                                               │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[22:10:26] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 81.82% @ 100                                                                                                 
           INFO     Ablating LKG model for next loop (mode: full)...                                                                                                       
           INFO     🧠 (Full Mode) Partially ablating neuron 3 in layer 'layer_stack.3.blocks.6.layers.0'.                                                                 

Meta-Loops ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100/100 0:00:00
           INFO     Final LKG model stored at: models/mnist_lkg.safetensors                                                                                                
           INFO     🏆 Final Bounty (best validation accuracy achieved): 81.82% @ 100/100                                                                                  

[22:10:27] INFO     🧪 Final Test Accuracy: 82.56%                                                                                                                         
╭──────────────────────────────────────────────────────────────────────────────────╮
│ ✅ Training Finished. Final Bounty (Validation): 81.82% @ 100/100 | Test: 82.56% │
╰──────────────────────────────────────────────────────────────────────────────────╯

## 18*18 Manual Tests (Single Trials of All Modes, with 1 Full Trial Output Recorded)

None: 95.59% @ 99
Decay: 95.92% @ 96
Dropout: 94.81 @ 78
Full: 95.74% @ 63
Hidden: 96.03% @ 100
Output: 95.74% @ 97

(src-py3.11) (base) timcotten@Tims-Laptop-2 002-persistent-stochastic-ablation-resmlp % poetry run clean; poetry run train --arch="[18*18]" --ablation-mode hidden
Warning: 'clean' is an entry point defined in pyproject.toml, but it's not installed as a script. You may get improper `sys.argv[0]`.

The support to run uninstalled scripts will be removed in a future release.

Run `poetry install` to resolve and get rid of this message.

                     Files to Remove                     
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ File Path                    ┃          Size ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ models/mnist_lkg.safetensors │ 136,099 bytes │ Exists │
└──────────────────────────────┴───────────────┴────────┘
Total size to remove: 136,099 bytes
╭──────────────────────────────────────────────────────────────────────────── Clean Complete ─────────────────────────────────────────────────────────────────────────────╮
│ ✅ Successfully removed 1 file(s)                                                                                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Warning: 'train' is an entry point defined in pyproject.toml, but it's not installed as a script. You may get improper `sys.argv[0]`.

The support to run uninstalled scripts will be removed in a future release.

Run `poetry install` to resolve and get rid of this message.

[23:39:50] INFO     Ablator (hidden mode) identified 36 hidden layers for ablation.                                                                                        
           INFO     Ablator (hidden mode) indexed 648 hidden neurons.                                                                                                      
╭─────────────────────────────────────────╮
│ Frustration Engine: MNIST Ablation Test │
╰── Ablation Mode: hidden | Device: mps ──╯
                         ResMLP Architecture Summary: [18*18]                          
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃   Metric   ┃ Initial Proj. ┃ ResStack (18x18) ┃ Final Proj. ┃   ┃         ┃         ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━╇━━━━━━━━━╇━━━━━━━━━┩
│   Shape    │      784      │     784 → 18     │      ✓      │ ✓ │ 18 → 18 │ 18 → 10 │
│ Parameters │       -       │      14,130      │      -      │ - │ 12,312  │ 190     │
└────────────┴───────────────┴──────────────────┴─────────────┴───┴─────────┴─────────┘
[23:39:54] INFO     Model has 26,632 parameters.                                                                                                                           
           INFO     Training for 100 meta-loops.                                                                                                                           
           INFO     📊 Dataset splits: 50k train, 10k validation, 10k test                                                                                                 
           INFO     🎯 Meta-loops use validation accuracy for LKG decisions                                                                                                
           INFO     🧪 Final test accuracy reported at completion                                                                                                          
           WARNING  No checkpoint found at models/mnist_lkg.safetensors. Starting from scratch.                                                                            
╭────────────────────────────────────────────────────────────────────── Meta-Loop 1/100 (Global: 1) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    -1.00%                                                                                                                            │
│   Current Loop Validation Accuracy    76.79%                                                                                                                            │
│   Global Meta-Loop                    1                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:39:58] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 76.79% @ 1                                                                                                   
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 3 in hidden layer 'layer_stack.3.blocks.2.layers.0'.                                                            

╭────────────────────────────────────────────────────────────────────── Meta-Loop 2/100 (Global: 2) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    76.79%                                                                                                                            │
│   Current Loop Validation Accuracy    85.96%                                                                                                                            │
│   Global Meta-Loop                    2                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:02] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 85.96% @ 2                                                                                                   
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 17 in hidden layer 'layer_stack.3.blocks.14.layers.0'.                                                          

╭────────────────────────────────────────────────────────────────────── Meta-Loop 3/100 (Global: 3) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    85.96%                                                                                                                            │
│   Current Loop Validation Accuracy    89.01%                                                                                                                            │
│   Global Meta-Loop                    3                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:05] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 89.01% @ 3                                                                                                   
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 15 in hidden layer 'layer_stack.3.blocks.0.layers.3'.                                                           

╭────────────────────────────────────────────────────────────────────── Meta-Loop 4/100 (Global: 4) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    89.01%                                                                                                                            │
│   Current Loop Validation Accuracy    90.13%                                                                                                                            │
│   Global Meta-Loop                    4                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:09] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 90.13% @ 4                                                                                                   
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 11 in hidden layer 'layer_stack.3.blocks.2.layers.0'.                                                           

╭────────────────────────────────────────────────────────────────────── Meta-Loop 5/100 (Global: 5) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    90.13%                                                                                                                            │
│   Current Loop Validation Accuracy    91.13%                                                                                                                            │
│   Global Meta-Loop                    5                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:12] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 91.13% @ 5                                                                                                   
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 16 in hidden layer 'layer_stack.3.blocks.6.layers.0'.                                                           

╭────────────────────────────────────────────────────────────────────── Meta-Loop 6/100 (Global: 6) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    91.13%                                                                                                                            │
│   Current Loop Validation Accuracy    91.88%                                                                                                                            │
│   Global Meta-Loop                    6                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:16] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 91.88% @ 6                                                                                                   
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 8 in hidden layer 'layer_stack.3.blocks.17.layers.3'.                                                           

╭────────────────────────────────────────────────────────────────────── Meta-Loop 7/100 (Global: 7) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    91.88%                                                                                                                            │
│   Current Loop Validation Accuracy    92.48%                                                                                                                            │
│   Global Meta-Loop                    7                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:19] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 92.48% @ 7                                                                                                   
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 15 in hidden layer 'layer_stack.3.blocks.4.layers.0'.                                                           

╭────────────────────────────────────────────────────────────────────── Meta-Loop 8/100 (Global: 8) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    92.48%                                                                                                                            │
│   Current Loop Validation Accuracy    92.97%                                                                                                                            │
│   Global Meta-Loop                    8                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:23] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 92.97% @ 8                                                                                                   
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 10 in hidden layer 'layer_stack.3.blocks.17.layers.0'.                                                          

╭────────────────────────────────────────────────────────────────────── Meta-Loop 9/100 (Global: 9) ──────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    92.97%                                                                                                                            │
│   Current Loop Validation Accuracy    93.24%                                                                                                                            │
│   Global Meta-Loop                    9                                                                                                                                 │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:27] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 93.24% @ 9                                                                                                   
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 5 in hidden layer 'layer_stack.3.blocks.14.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 10/100 (Global: 10) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    93.24%                                                                                                                            │
│   Current Loop Validation Accuracy    93.63%                                                                                                                            │
│   Global Meta-Loop                    10                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:30] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 93.63% @ 10                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 2 in hidden layer 'layer_stack.3.blocks.17.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 11/100 (Global: 11) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    93.63%                                                                                                                            │
│   Current Loop Validation Accuracy    93.88%                                                                                                                            │
│   Global Meta-Loop                    11                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:34] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 93.88% @ 11                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 2 in hidden layer 'layer_stack.3.blocks.11.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 12/100 (Global: 12) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    93.88%                                                                                                                            │
│   Current Loop Validation Accuracy    94.06%                                                                                                                            │
│   Global Meta-Loop                    12                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:38] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 94.06% @ 12                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 8 in hidden layer 'layer_stack.3.blocks.13.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 13/100 (Global: 13) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.06%                                                                                                                            │
│   Current Loop Validation Accuracy    94.20%                                                                                                                            │
│   Global Meta-Loop                    13                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:42] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 94.20% @ 13                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 12 in hidden layer 'layer_stack.3.blocks.5.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 14/100 (Global: 14) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.20%                                                                                                                            │
│   Current Loop Validation Accuracy    94.03%                                                                                                                            │
│   Global Meta-Loop                    14                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:40:46] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 3 in hidden layer 'layer_stack.3.blocks.13.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 15/100 (Global: 15) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.20%                                                                                                                            │
│   Current Loop Validation Accuracy    94.25%                                                                                                                            │
│   Global Meta-Loop                    15                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:49] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 94.25% @ 15                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 2 in hidden layer 'layer_stack.3.blocks.1.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 16/100 (Global: 16) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.25%                                                                                                                            │
│   Current Loop Validation Accuracy    94.25%                                                                                                                            │
│   Global Meta-Loop                    16                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:40:53] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 6 in hidden layer 'layer_stack.3.blocks.10.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 17/100 (Global: 17) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.25%                                                                                                                            │
│   Current Loop Validation Accuracy    94.27%                                                                                                                            │
│   Global Meta-Loop                    17                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:40:57] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 94.27% @ 17                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 7 in hidden layer 'layer_stack.3.blocks.6.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 18/100 (Global: 18) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.27%                                                                                                                            │
│   Current Loop Validation Accuracy    94.44%                                                                                                                            │
│   Global Meta-Loop                    18                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:41:00] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 94.44% @ 18                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 16 in hidden layer 'layer_stack.3.blocks.7.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 19/100 (Global: 19) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.44%                                                                                                                            │
│   Current Loop Validation Accuracy    94.50%                                                                                                                            │
│   Global Meta-Loop                    19                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:41:04] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 94.50% @ 19                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 3 in hidden layer 'layer_stack.3.blocks.12.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 20/100 (Global: 20) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.50%                                                                                                                            │
│   Current Loop Validation Accuracy    94.68%                                                                                                                            │
│   Global Meta-Loop                    20                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:41:07] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 94.68% @ 20                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 13 in hidden layer 'layer_stack.3.blocks.10.layers.0'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 21/100 (Global: 21) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.68%                                                                                                                            │
│   Current Loop Validation Accuracy    94.78%                                                                                                                            │
│   Global Meta-Loop                    21                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:41:11] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 94.78% @ 21                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 4 in hidden layer 'layer_stack.3.blocks.1.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 22/100 (Global: 22) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.78%                                                                                                                            │
│   Current Loop Validation Accuracy    94.82%                                                                                                                            │
│   Global Meta-Loop                    22                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:41:14] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 94.82% @ 22                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 2 in hidden layer 'layer_stack.3.blocks.4.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 23/100 (Global: 23) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.82%                                                                                                                            │
│   Current Loop Validation Accuracy    94.77%                                                                                                                            │
│   Global Meta-Loop                    23                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:41:18] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 9 in hidden layer 'layer_stack.3.blocks.0.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 24/100 (Global: 24) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.82%                                                                                                                            │
│   Current Loop Validation Accuracy    94.74%                                                                                                                            │
│   Global Meta-Loop                    24                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:41:22] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 10 in hidden layer 'layer_stack.3.blocks.16.layers.0'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 25/100 (Global: 25) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.82%                                                                                                                            │
│   Current Loop Validation Accuracy    94.86%                                                                                                                            │
│   Global Meta-Loop                    25                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:41:25] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 94.86% @ 25                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 2 in hidden layer 'layer_stack.3.blocks.1.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 26/100 (Global: 26) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.86%                                                                                                                            │
│   Current Loop Validation Accuracy    94.94%                                                                                                                            │
│   Global Meta-Loop                    26                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:41:29] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 94.94% @ 26                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 4 in hidden layer 'layer_stack.3.blocks.6.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 27/100 (Global: 27) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.94%                                                                                                                            │
│   Current Loop Validation Accuracy    94.87%                                                                                                                            │
│   Global Meta-Loop                    27                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:41:32] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 17 in hidden layer 'layer_stack.3.blocks.15.layers.3'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 28/100 (Global: 28) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.94%                                                                                                                            │
│   Current Loop Validation Accuracy    94.92%                                                                                                                            │
│   Global Meta-Loop                    28                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:41:36] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 5 in hidden layer 'layer_stack.3.blocks.16.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 29/100 (Global: 29) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.94%                                                                                                                            │
│   Current Loop Validation Accuracy    94.94%                                                                                                                            │
│   Global Meta-Loop                    29                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:41:39] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 9 in hidden layer 'layer_stack.3.blocks.16.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 30/100 (Global: 30) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.94%                                                                                                                            │
│   Current Loop Validation Accuracy    94.97%                                                                                                                            │
│   Global Meta-Loop                    30                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:41:43] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 94.97% @ 30                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 3 in hidden layer 'layer_stack.3.blocks.2.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 31/100 (Global: 31) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.97%                                                                                                                            │
│   Current Loop Validation Accuracy    94.93%                                                                                                                            │
│   Global Meta-Loop                    31                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:41:47] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 17 in hidden layer 'layer_stack.3.blocks.4.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 32/100 (Global: 32) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.97%                                                                                                                            │
│   Current Loop Validation Accuracy    94.86%                                                                                                                            │
│   Global Meta-Loop                    32                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:41:50] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 5 in hidden layer 'layer_stack.3.blocks.10.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 33/100 (Global: 33) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.97%                                                                                                                            │
│   Current Loop Validation Accuracy    94.97%                                                                                                                            │
│   Global Meta-Loop                    33                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:41:54] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 4 in hidden layer 'layer_stack.3.blocks.16.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 34/100 (Global: 34) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    94.97%                                                                                                                            │
│   Current Loop Validation Accuracy    95.06%                                                                                                                            │
│   Global Meta-Loop                    34                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:41:57] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.06% @ 34                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
[23:41:58] INFO     🧠 (Hidden Mode) Fully ablating neuron 13 in hidden layer 'layer_stack.3.blocks.4.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 35/100 (Global: 35) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.06%                                                                                                                            │
│   Current Loop Validation Accuracy    95.09%                                                                                                                            │
│   Global Meta-Loop                    35                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:42:01] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.09% @ 35                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 7 in hidden layer 'layer_stack.3.blocks.11.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 36/100 (Global: 36) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.09%                                                                                                                            │
│   Current Loop Validation Accuracy    95.19%                                                                                                                            │
│   Global Meta-Loop                    36                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:42:05] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.19% @ 36                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 6 in hidden layer 'layer_stack.3.blocks.0.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 37/100 (Global: 37) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.19%                                                                                                                            │
│   Current Loop Validation Accuracy    95.22%                                                                                                                            │
│   Global Meta-Loop                    37                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:42:08] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.22% @ 37                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 13 in hidden layer 'layer_stack.3.blocks.2.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 38/100 (Global: 38) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.22%                                                                                                                            │
│   Current Loop Validation Accuracy    95.34%                                                                                                                            │
│   Global Meta-Loop                    38                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:42:12] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.34% @ 38                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 15 in hidden layer 'layer_stack.3.blocks.17.layers.0'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 39/100 (Global: 39) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.34%                                                                                                                            │
│   Current Loop Validation Accuracy    95.20%                                                                                                                            │
│   Global Meta-Loop                    39                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:42:15] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 9 in hidden layer 'layer_stack.3.blocks.12.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 40/100 (Global: 40) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.34%                                                                                                                            │
│   Current Loop Validation Accuracy    95.39%                                                                                                                            │
│   Global Meta-Loop                    40                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:42:19] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.39% @ 40                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 0 in hidden layer 'layer_stack.3.blocks.16.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 41/100 (Global: 41) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.39%                                                                                                                            │
│   Current Loop Validation Accuracy    95.31%                                                                                                                            │
│   Global Meta-Loop                    41                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:42:23] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 2 in hidden layer 'layer_stack.3.blocks.3.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 42/100 (Global: 42) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.39%                                                                                                                            │
│   Current Loop Validation Accuracy    95.43%                                                                                                                            │
│   Global Meta-Loop                    42                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:42:26] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.43% @ 42                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 8 in hidden layer 'layer_stack.3.blocks.8.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 43/100 (Global: 43) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.43%                                                                                                                            │
│   Current Loop Validation Accuracy    95.54%                                                                                                                            │
│   Global Meta-Loop                    43                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:42:30] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.54% @ 43                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 7 in hidden layer 'layer_stack.3.blocks.4.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 44/100 (Global: 44) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.45%                                                                                                                            │
│   Global Meta-Loop                    44                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:42:33] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 11 in hidden layer 'layer_stack.3.blocks.10.layers.0'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 45/100 (Global: 45) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.53%                                                                                                                            │
│   Global Meta-Loop                    45                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:42:37] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 0 in hidden layer 'layer_stack.3.blocks.9.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 46/100 (Global: 46) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.36%                                                                                                                            │
│   Global Meta-Loop                    46                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:42:40] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 5 in hidden layer 'layer_stack.3.blocks.12.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 47/100 (Global: 47) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.41%                                                                                                                            │
│   Global Meta-Loop                    47                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:42:44] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 14 in hidden layer 'layer_stack.3.blocks.10.layers.0'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 48/100 (Global: 48) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.36%                                                                                                                            │
│   Global Meta-Loop                    48                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:42:48] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 0 in hidden layer 'layer_stack.3.blocks.12.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 49/100 (Global: 49) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.44%                                                                                                                            │
│   Global Meta-Loop                    49                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:42:51] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 11 in hidden layer 'layer_stack.3.blocks.4.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 50/100 (Global: 50) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.36%                                                                                                                            │
│   Global Meta-Loop                    50                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:42:55] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 0 in hidden layer 'layer_stack.3.blocks.15.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 51/100 (Global: 51) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.51%                                                                                                                            │
│   Global Meta-Loop                    51                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:42:59] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 9 in hidden layer 'layer_stack.3.blocks.10.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 52/100 (Global: 52) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.47%                                                                                                                            │
│   Global Meta-Loop                    52                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:02] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 16 in hidden layer 'layer_stack.3.blocks.7.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 53/100 (Global: 53) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.54%                                                                                                                            │
│   Global Meta-Loop                    53                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:06] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 8 in hidden layer 'layer_stack.3.blocks.15.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 54/100 (Global: 54) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.42%                                                                                                                            │
│   Global Meta-Loop                    54                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:09] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
[23:43:10] INFO     🧠 (Hidden Mode) Fully ablating neuron 3 in hidden layer 'layer_stack.3.blocks.9.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 55/100 (Global: 55) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.46%                                                                                                                            │
│   Global Meta-Loop                    55                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:13] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 9 in hidden layer 'layer_stack.3.blocks.4.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 56/100 (Global: 56) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.50%                                                                                                                            │
│   Global Meta-Loop                    56                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:17] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 16 in hidden layer 'layer_stack.3.blocks.16.layers.3'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 57/100 (Global: 57) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.54%                                                                                                                            │
│   Current Loop Validation Accuracy    95.55%                                                                                                                            │
│   Global Meta-Loop                    57                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:43:20] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.55% @ 57                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 3 in hidden layer 'layer_stack.3.blocks.11.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 58/100 (Global: 58) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.55%                                                                                                                            │
│   Current Loop Validation Accuracy    95.47%                                                                                                                            │
│   Global Meta-Loop                    58                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:24] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 15 in hidden layer 'layer_stack.3.blocks.1.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 59/100 (Global: 59) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.55%                                                                                                                            │
│   Current Loop Validation Accuracy    95.43%                                                                                                                            │
│   Global Meta-Loop                    59                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:28] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 13 in hidden layer 'layer_stack.3.blocks.5.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 60/100 (Global: 60) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.55%                                                                                                                            │
│   Current Loop Validation Accuracy    95.46%                                                                                                                            │
│   Global Meta-Loop                    60                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:31] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 16 in hidden layer 'layer_stack.3.blocks.17.layers.0'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 61/100 (Global: 61) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.55%                                                                                                                            │
│   Current Loop Validation Accuracy    95.42%                                                                                                                            │
│   Global Meta-Loop                    61                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:35] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 5 in hidden layer 'layer_stack.3.blocks.14.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 62/100 (Global: 62) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.55%                                                                                                                            │
│   Current Loop Validation Accuracy    95.54%                                                                                                                            │
│   Global Meta-Loop                    62                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:39] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 9 in hidden layer 'layer_stack.3.blocks.8.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 63/100 (Global: 63) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.55%                                                                                                                            │
│   Current Loop Validation Accuracy    95.47%                                                                                                                            │
│   Global Meta-Loop                    63                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:42] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 3 in hidden layer 'layer_stack.3.blocks.15.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 64/100 (Global: 64) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.55%                                                                                                                            │
│   Current Loop Validation Accuracy    95.47%                                                                                                                            │
│   Global Meta-Loop                    64                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:46] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 12 in hidden layer 'layer_stack.3.blocks.4.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 65/100 (Global: 65) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.55%                                                                                                                            │
│   Current Loop Validation Accuracy    95.49%                                                                                                                            │
│   Global Meta-Loop                    65                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:50] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 13 in hidden layer 'layer_stack.3.blocks.11.layers.3'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 66/100 (Global: 66) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.55%                                                                                                                            │
│   Current Loop Validation Accuracy    95.40%                                                                                                                            │
│   Global Meta-Loop                    66                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:53] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 6 in hidden layer 'layer_stack.3.blocks.2.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 67/100 (Global: 67) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.55%                                                                                                                            │
│   Current Loop Validation Accuracy    95.52%                                                                                                                            │
│   Global Meta-Loop                    67                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:43:57] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 12 in hidden layer 'layer_stack.3.blocks.4.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 68/100 (Global: 68) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.55%                                                                                                                            │
│   Current Loop Validation Accuracy    95.47%                                                                                                                            │
│   Global Meta-Loop                    68                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:44:00] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
[23:44:01] INFO     🧠 (Hidden Mode) Fully ablating neuron 2 in hidden layer 'layer_stack.3.blocks.4.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 69/100 (Global: 69) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.55%                                                                                                                            │
│   Current Loop Validation Accuracy    95.51%                                                                                                                            │
│   Global Meta-Loop                    69                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:44:04] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 10 in hidden layer 'layer_stack.3.blocks.5.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 70/100 (Global: 70) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.55%                                                                                                                            │
│   Current Loop Validation Accuracy    95.56%                                                                                                                            │
│   Global Meta-Loop                    70                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:44:08] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.56% @ 70                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 16 in hidden layer 'layer_stack.3.blocks.14.layers.3'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 71/100 (Global: 71) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.56%                                                                                                                            │
│   Current Loop Validation Accuracy    95.60%                                                                                                                            │
│   Global Meta-Loop                    71                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:44:12] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.60% @ 71                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 12 in hidden layer 'layer_stack.3.blocks.0.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 72/100 (Global: 72) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.60%                                                                                                                            │
│   Current Loop Validation Accuracy    95.55%                                                                                                                            │
│   Global Meta-Loop                    72                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:44:15] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 13 in hidden layer 'layer_stack.3.blocks.6.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 73/100 (Global: 73) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.60%                                                                                                                            │
│   Current Loop Validation Accuracy    95.52%                                                                                                                            │
│   Global Meta-Loop                    73                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:44:19] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 6 in hidden layer 'layer_stack.3.blocks.6.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 74/100 (Global: 74) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.60%                                                                                                                            │
│   Current Loop Validation Accuracy    95.62%                                                                                                                            │
│   Global Meta-Loop                    74                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:44:23] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.62% @ 74                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 7 in hidden layer 'layer_stack.3.blocks.8.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 75/100 (Global: 75) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.62%                                                                                                                            │
│   Current Loop Validation Accuracy    95.58%                                                                                                                            │
│   Global Meta-Loop                    75                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:44:27] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 1 in hidden layer 'layer_stack.3.blocks.5.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 76/100 (Global: 76) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.62%                                                                                                                            │
│   Current Loop Validation Accuracy    95.52%                                                                                                                            │
│   Global Meta-Loop                    76                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:44:30] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 12 in hidden layer 'layer_stack.3.blocks.15.layers.3'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 77/100 (Global: 77) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.62%                                                                                                                            │
│   Current Loop Validation Accuracy    95.66%                                                                                                                            │
│   Global Meta-Loop                    77                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:44:34] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.66% @ 77                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 8 in hidden layer 'layer_stack.3.blocks.14.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 78/100 (Global: 78) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.66%                                                                                                                            │
│   Current Loop Validation Accuracy    95.66%                                                                                                                            │
│   Global Meta-Loop                    78                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:44:38] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 0 in hidden layer 'layer_stack.3.blocks.2.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 79/100 (Global: 79) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.66%                                                                                                                            │
│   Current Loop Validation Accuracy    95.67%                                                                                                                            │
│   Global Meta-Loop                    79                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:44:41] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.67% @ 79                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 6 in hidden layer 'layer_stack.3.blocks.1.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 80/100 (Global: 80) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.67%                                                                                                                            │
│   Current Loop Validation Accuracy    95.74%                                                                                                                            │
│   Global Meta-Loop                    80                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:44:45] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.74% @ 80                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 9 in hidden layer 'layer_stack.3.blocks.8.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 81/100 (Global: 81) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.74%                                                                                                                            │
│   Current Loop Validation Accuracy    95.71%                                                                                                                            │
│   Global Meta-Loop                    81                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:44:48] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 17 in hidden layer 'layer_stack.3.blocks.8.layers.0'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 82/100 (Global: 82) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.74%                                                                                                                            │
│   Current Loop Validation Accuracy    95.65%                                                                                                                            │
│   Global Meta-Loop                    82                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:44:52] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 11 in hidden layer 'layer_stack.3.blocks.11.layers.3'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 83/100 (Global: 83) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.74%                                                                                                                            │
│   Current Loop Validation Accuracy    95.73%                                                                                                                            │
│   Global Meta-Loop                    83                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:44:56] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 9 in hidden layer 'layer_stack.3.blocks.8.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 84/100 (Global: 84) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.74%                                                                                                                            │
│   Current Loop Validation Accuracy    95.64%                                                                                                                            │
│   Global Meta-Loop                    84                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:44:59] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 2 in hidden layer 'layer_stack.3.blocks.5.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 85/100 (Global: 85) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.74%                                                                                                                            │
│   Current Loop Validation Accuracy    95.78%                                                                                                                            │
│   Global Meta-Loop                    85                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:45:03] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.78% @ 85                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 0 in hidden layer 'layer_stack.3.blocks.1.layers.0'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 86/100 (Global: 86) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.78%                                                                                                                            │
│   Current Loop Validation Accuracy    95.79%                                                                                                                            │
│   Global Meta-Loop                    86                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:45:07] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.79% @ 86                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 1 in hidden layer 'layer_stack.3.blocks.8.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 87/100 (Global: 87) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.79%                                                                                                                            │
│   Current Loop Validation Accuracy    95.82%                                                                                                                            │
│   Global Meta-Loop                    87                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:45:10] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.82% @ 87                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 8 in hidden layer 'layer_stack.3.blocks.4.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 88/100 (Global: 88) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.82%                                                                                                                            │
│   Current Loop Validation Accuracy    95.82%                                                                                                                            │
│   Global Meta-Loop                    88                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:45:14] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 15 in hidden layer 'layer_stack.3.blocks.16.layers.3'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 89/100 (Global: 89) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.82%                                                                                                                            │
│   Current Loop Validation Accuracy    95.82%                                                                                                                            │
│   Global Meta-Loop                    89                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:45:18] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 16 in hidden layer 'layer_stack.3.blocks.4.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 90/100 (Global: 90) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.82%                                                                                                                            │
│   Current Loop Validation Accuracy    95.77%                                                                                                                            │
│   Global Meta-Loop                    90                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:45:21] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 1 in hidden layer 'layer_stack.3.blocks.15.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 91/100 (Global: 91) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.82%                                                                                                                            │
│   Current Loop Validation Accuracy    95.78%                                                                                                                            │
│   Global Meta-Loop                    91                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:45:25] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 0 in hidden layer 'layer_stack.3.blocks.14.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 92/100 (Global: 92) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.82%                                                                                                                            │
│   Current Loop Validation Accuracy    95.72%                                                                                                                            │
│   Global Meta-Loop                    92                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:45:28] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
[23:45:29] INFO     🧠 (Hidden Mode) Fully ablating neuron 14 in hidden layer 'layer_stack.3.blocks.3.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 93/100 (Global: 93) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.82%                                                                                                                            │
│   Current Loop Validation Accuracy    95.72%                                                                                                                            │
│   Global Meta-Loop                    93                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:45:32] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 12 in hidden layer 'layer_stack.3.blocks.11.layers.3'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 94/100 (Global: 94) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.82%                                                                                                                            │
│   Current Loop Validation Accuracy    95.88%                                                                                                                            │
│   Global Meta-Loop                    94                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:45:36] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.88% @ 94                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 2 in hidden layer 'layer_stack.3.blocks.13.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 95/100 (Global: 95) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.88%                                                                                                                            │
│   Current Loop Validation Accuracy    95.88%                                                                                                                            │
│   Global Meta-Loop                    95                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:45:39] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 5 in hidden layer 'layer_stack.3.blocks.12.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 96/100 (Global: 96) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.88%                                                                                                                            │
│   Current Loop Validation Accuracy    95.71%                                                                                                                            │
│   Global Meta-Loop                    96                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:45:43] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 17 in hidden layer 'layer_stack.3.blocks.3.layers.3'.                                                           

╭───────────────────────────────────────────────────────────────────── Meta-Loop 97/100 (Global: 97) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.88%                                                                                                                            │
│   Current Loop Validation Accuracy    95.89%                                                                                                                            │
│   Global Meta-Loop                    97                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:45:46] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.89% @ 97                                                                                                  
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 1 in hidden layer 'layer_stack.3.blocks.2.layers.3'.                                                            

╭───────────────────────────────────────────────────────────────────── Meta-Loop 98/100 (Global: 98) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.89%                                                                                                                            │
│   Current Loop Validation Accuracy    95.86%                                                                                                                            │
│   Global Meta-Loop                    98                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:45:50] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 10 in hidden layer 'layer_stack.3.blocks.10.layers.0'.                                                          

╭───────────────────────────────────────────────────────────────────── Meta-Loop 99/100 (Global: 99) ─────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.89%                                                                                                                            │
│   Current Loop Validation Accuracy    95.83%                                                                                                                            │
│   Global Meta-Loop                    99                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────── NO IMPROVEMENT ─────────────────────────────────────────────────────────────────────────────╯
[23:45:54] INFO     Discarding weights.                                                                                                                                    
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 3 in hidden layer 'layer_stack.3.blocks.17.layers.0'.                                                           

╭──────────────────────────────────────────────────────────────────── Meta-Loop 100/100 (Global: 100) ────────────────────────────────────────────────────────────────────╮
│   Previous LKG Validation Accuracy    95.89%                                                                                                                            │
│   Current Loop Validation Accuracy    95.91%                                                                                                                            │
│   Global Meta-Loop                    100                                                                                                                               │
╰────────────────────────────────────────────────────────────────────────────── IMPROVEMENT ──────────────────────────────────────────────────────────────────────────────╯
[23:45:57] INFO     New LKG. Checkpoint saved. 🏆 New Bounty: 95.91% @ 100                                                                                                 
           INFO     Ablating LKG model for next loop (mode: hidden)...                                                                                                     
           INFO     🧠 (Hidden Mode) Fully ablating neuron 7 in hidden layer 'layer_stack.3.blocks.3.layers.3'.                                                            

Meta-Loops ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100/100 0:00:00
           INFO     Final LKG model stored at: models/mnist_lkg.safetensors                                                                                                
           INFO     🏆 Final Bounty (best validation accuracy achieved): 95.91% @ 100/100                                                                                  

           INFO     🧪 Final Test Accuracy: 96.03%                                                                                                                         
╭──────────────────────────────────────────────────────────────────────────────────╮
│ ✅ Training Finished. Final Bounty (Validation): 95.91% @ 100/100 | Test: 96.03% │
╰──────────────────────────────────────────────────────────────────────────────────╯

