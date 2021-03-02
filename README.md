![Pet Banner](images/header.png)
# <center> Predicting Pet Adoption Time Using **LightGBM Gradient Boosting**</center>

<br>

The goal of this project is to develop an algorithm to predict the adoptability of pets - specifically, how quickly is a pet adopted? The dataset was part of the [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction).

<br>

**Model Accuracy**: (Cohen's Kappa = 0.439)

    Top submissions on the public leaderboard are around 0.45


**Competition Description**
> Millions of stray animals suffer on the streets or are euthanized in shelters every day around the world. If homes can be found for them, many precious lives can be saved — and more happy families created.
<img style="float: right;" src="images/thumb76_76.png">

> PetFinder.my has been Malaysia’s leading animal welfare platform since 2008, with a database of more than 150,000 animals. PetFinder collaborates closely with animal lovers, media, corporations, and global organizations to improve animal welfare.
>
> Animal adoption rates are strongly correlated to the metadata associated with their online profiles, such as descriptive text and photo characteristics. As one example, PetFinder is currently experimenting with a simple AI tool called the Cuteness Meter, which ranks how cute a pet is based on qualities present in their photos.
>
> In this competition you will be developing algorithms to predict the adoptability of pets - specifically, how quickly is a pet adopted? If successful, they will be adapted into AI tools that will guide shelters and rescuers around the world on improving their pet profiles' appeal, reducing animal suffering and euthanization.
>
> Top participants may be invited to collaborate on implementing their solutions into AI tools for assessing and improving pet adoption performance, which will benefit global animal welfare.

**Competition Evaluation**
> Submissions are scored based on the quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters).
>
> In the event that there is less agreement between the raters than expected by chance, the metric may go below 0. The quadratic weighted kappa is calculated between the scores which are expected/known and the predicted scores.
