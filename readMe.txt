This is the repository for PF2PI, a protein function prediction method based on AlphaFold2 and PPI


annotation_preprocess2.py is used to prepocess Gene Ontology annotation file in terms of STRING PPI network data. 

data_net.py is used to prepocess STRING PPI network data to weighted adjacency matrix.

protein_struct_map.py is used to get protein contact map.

node2vec.py is used to get random walk path.

screen.py is used to select the proteins present in both databases.

get_embeddings_struct_features.py is used to get the final sturcture features.

pre_trainer.py is used to pre-train CFAGO in terms of reconstruct protein features.

fine-tuning.py is used to fine-tune CFAGO and predict functions for testing proteins.

Uses pytorch 1.12.1

---------- Sample Use Case ----------

Let's say you want to conduct experiments on human dataset.

Here is what you need to run:

Step 1:

Preprocess STRING PPI file, structure file.

python data_annotation_2.py -data_path Dataset -af goa_human.gaf -pf 9606.protein.info.v11.5.txt -ppif 9606.protein.links.detailed.v11.5.txt -org human -stl 41

python data_net.py -data_path Dataset -ppif 9606.protein.links.detailed.v11.5.txt -org human

python protein_struct_map.py

python node2vec.py

python screen.py

python get_embeddings_struct_features.py

Step 2:

Pre-train by self-supervised learning:

python pre_trainer.py --org human --dataset_dir Dataset/human --output human_result --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5

python pre_trainer.py --org human --dataset_dir Dataset/human --output human_result_ablation --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5

Step 3:

fine-tuning CFAGO with annotations as labels, and output predictions for test proteins in terms of one GO branch: P for biological process ontology, F for moleculer function ontology, C for cellular component  ontology.

python fine-tuning.py --org human --dataset_dir Dataset/human --output human_result --aspect P --num_class 45 --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.3 --attention_layers 6 --gamma_pos 0 --gamma_neg 2 --batch-size 32 --activation gelu --lr 1e-4 --pretrained_model human_result/human_attention_layers_6_lr_1e-05_seed_1329765522_activation_gelu_model.pkl

python fine-tuning.py --org human --dataset_dir Dataset/human --output human_result --aspect F --num_class 38 --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.3 --attention_layers 6 --gamma_pos 0 --gamma_neg 2 --batch-size 32 --activation gelu --lr 1e-4 --pretrained_model human_result/human_attention_layers_6_lr_1e-05_seed_1329765522_activation_gelu_model.pkl

python fine-tuning.py --org human --dataset_dir Dataset/human --output human_result --aspect C --num_class 35 --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.3 --attention_layers 6 --gamma_pos 0 --gamma_neg 2 --batch-size 32 --activation gelu --lr 1e-4 --pretrained_model human_result/human_attention_layers_6_lr_1e-05_seed_1329765522_activation_gelu_model.pkl

The results file that this step produces will be found at this path: ./human_result/human_attention_layers_6_aspect_P_fintune_seed_1329765522_act_gelu.csv. This file contains the five evaluation matrices' values on the Biological Process Ontology.

python fine-tuning.py --org human --dataset_dir Dataset/human --output human_result_ablation --aspect P --num_class 45 --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.3 --attention_layers 6 --gamma_pos 0 --gamma_neg 2 --batch-size 32 --activation gelu --lr 1e-4 --pretrained_model human_result_ablation/human_attention_layers_6_lr_1e-05_seed_1329765522_activation_gelu_model.pkl

python fine-tuning.py --org human --dataset_dir Dataset/human --output human_result_ablation --aspect F --num_class 38 --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.3 --attention_layers 6 --gamma_pos 0 --gamma_neg 2 --batch-size 32 --activation gelu --lr 1e-4 --pretrained_model human_result_ablation/human_attention_layers_6_lr_1e-05_seed_1329765522_activation_gelu_model.pkl

python fine-tuning.py --org human --dataset_dir Dataset/human --output human_result_ablation --aspect C --num_class 35 --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.3 --attention_layers 6 --gamma_pos 0 --gamma_neg 2 --batch-size 32 --activation gelu --lr 1e-4 --pretrained_model human_result_ablation/human_attention_layers_6_lr_1e-05_seed_1329765522_activation_gelu_model.pkl
