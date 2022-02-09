# Training privacy-preserving video analytics pipelines by suppressing features that reveal information about private attributes
<p align="center">
Chau Yi Li* and Andrea Cavallaro</br>
Queen Mary University of London, London, United Kingdom</br>
*chauyi.li@qmul.ac.uk</br>
</p>




## Abstract
Deep neural networks are increasingly deployed for scene analytics, including to evaluate the attention and reaction of people exposed to digital out-of-home advertisements. However,  the features in deep neural networks trained to predict a specific, consensual attribute such as attention may also encode and thus reveal private, protected attributes such as age or gender. In this work, we focus on such leakage of private information at inference time. We consider an adversary who have access to the features extracted by the layers of a deployed  neural network and use these features to predict private attributes. To prevent the success of this attack, we modify the training of the network using a confusion loss, which encourages the extraction of features that make it difficult for the adversary to accurately predict the private attributes. We validate this training approach on image-based tasks using a publicly available dataset. Results show that, compared to the original network,  the proposed strategy can reduce the leakage of private information from a state-of-the-art emotion recognition classifier by 2.88% for gender and and 13.06% for age group, with a minimal effect on the task accuracy. The paper has been acceptted in ICASSP 2022. 

## Train
- Requirements

  Torch 1.7.1, APEX 0.1, and torchvision 0.8.2.

- Data Preparation
  We have provided the labels for sensitive attributes (Gender, Age, Race) of the RAF-DB dataset. 

  Download [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset) dataset, and make sure it is saved in the datasets folder like this:
 
```
- datasets/raf-basic/
         EmoLabel/
             list_patition_label.txt
         Image/aligned/
	     train_00001_aligned.jpg
             test_0001_aligned.jpg
             ...
```
- Training original unprotected network
```
python train_raf-db.py
```

- Training adversary network
```
python train_raf-db_adv.py --checkpoint <unprotected network>.pth --attribute [age, gender]
```

- Training privacy-preserving network
```
python train_raf-db_adv.py --checkpoint *.pth --checkpoint <unprotected network>.pth --adversary <adversary network>.pth --attribute [age, gender]
```

# Citation
If you use the sample code or part of it in your research, please cite the following:

```
@ARTICLE{PrivacyPreserving_Pipeline_Li_Cavallaro_2022,
       author = {{Li}, C.Y. and {Cavallaro}, A.},
        title = "{Training privacy-preserving video analytics pipelines by suppressing features that reveal information about private attributes}",
      journal = {International Conference on Acoustics, Speech, and Signal Processing},
         year = 2022,
        month = May,
}
```

