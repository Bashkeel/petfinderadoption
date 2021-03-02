from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd

def transform_tfidf_svd(train, test):
    train_desc = train.Description.fillna("none").values
    test_desc = test.Description.fillna("none").values

    tfv = TfidfVectorizer(min_df=3, max_features=10000, strip_accents='unicode',
                         analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3),
                         use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')

    tfv.fit(list(train_desc))
    tfv_train =  tfv.transform(train_desc)
    print(f"train_X (tfidf): {tfv_train.shape}")

    tfv_test = tfv.transform(test_desc)
    print(f"test_X (tfidf): {tfv_test.shape}")



    # Dimensionality Reduction of the TF-IDF Vectorizer (n_components
    # corresponds to 50% of variance explained)
    # See eda-text-data.ipynb for explained variance by n_components
    n_components=646
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(tfv_train)
    svd_train = svd.transform(tfv_train)
    svd_test = svd.transform(tfv_test)
    print(f"train_X (svd): {svd_train.shape}")
    print(f"test_X (svd): {svd_test.shape}")


    train_svd_cols = pd.DataFrame(svd_train, columns=['svd_{}'.format(i) for i in range(n_components)])
    train = pd.concat((train, train_svd_cols), axis=1)

    test_svd_cols = pd.DataFrame(svd_test, columns=['svd_{}'.format(i) for i in range(n_components)])
    test = pd.concat((test, test_svd_cols), axis=1)

    return train, test
