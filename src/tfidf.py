
def select_n_components(var_ratio, goal_var: float) -> int:
    # Function taken from: https://chrisalbon.com/machine_learning/feature_engineering/select_best_number_of_components_in_tsvd/

    # Set initial variance explained so far
    total_variance = 0.0

    # Set initial number of features
    n_components = 0

    # For the explained variance of each feature:
    for explained_variance in var_ratio:

        # Add the explained variance to the total
        total_variance += explained_variance

        # Add one to the number of components
        n_components += 1

        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break

    # Return the number of components
    return n_components


def find_optimal_svd(df):
    tsvd = TruncatedSVD(n_components=tfv_train.shape[1]-1)
    TSVD_train = tsvd.fit(tfv_train)
    pickle.dump(TSVD_train, "N-1 TSVD Model.p")

    var_ratios = []
    for i in range(101):
        var_ratios.append(select_n_components(TSVD_train.explained_variance_ratio_, ((i+1)/100)))

    df_var = pd.DataFrame([list(range(101)), var_ratios]).transpose()
    df_var.columns = ['variance_explained', 'n_components']

    return df_var


def train_tfidf(df):
    train_desc = df.Description.fillna("none").values

    tfv = TfidfVectorizer(min_df=3, max_features=10000, strip_accents='unicode',
                         analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3),
                         use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')

    tfv.fit(list(train_desc))
    tfv_train =  tfv.transform(train_desc)
    print("X (tfidf):", tfv_train.shape)


    # Dimensionality Reduction of the TF-IDF Vectorizer (n_components
    # corresponds to 50% of variance explained)
    # See eda-text-data.ipynb for explained variance by n_components
    n_components=646
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(tfv_train)
    svd_train = svd.transform(tfv_train)
    print("X (svd):", svd_train.shape)

    svd_cols = pd.DataFrame(svd_train, columns=['svd_{}'.format(i) for i in range(len(svd_train))])
    df = pd.concat((df, svd_cols), axis=1)

    return df, tfv, svd


def transform_tfidf_svd(df, tfv, svd):
    df_desc = df.Description.fillna("none").values
    tfv_test = tfv.transform(df_desc)
    svd_test = svd.transform(tfv_test)
    svd_cols = pd.DataFrame(svd_test, columns=['svd_{}'.format(i) for i in range(len(svd_train))])
    df = pd.concat((df, svd_cols), axis=1)

    return df
