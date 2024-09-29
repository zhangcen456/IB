# Improving the Adversarial Robustness of NLP Models by Information Bottleneck
The code implemetation of ACL 2022 findings : Improving the Adversarial Robustness of NLP Models by Information Bottleneck

### train

```
python model.py -hd $hidden_dimension -be $beta -ai 0.0 -as 1
```

### combine with FreeLB

```
python model.py -hd $hidden_dimension -be $beta -ai $adv_init_mag -as $adv_steps -al $adv_lr
```

### attack

```
python model.py -hd $hidden_dimension -lp $model_path -a true -am $attack_method -ae $attack_example_num -mr $max_modify_ratio -ad $attack_dataset
```
