from imodelsx import EmbGAMClassifier
import pickle as pkl

# load model
with open(f"Data/huffpost.pkl", "rb") as f:
    huffpost_data = pkl.load(f)

# the keys 0, 1, 2 represent indistribution (ID) training and validation and out of distribution test set
# see table 6 for more info: https://arxiv.org/pdf/2211.14238.pdf
for year in [2012, 2013, 2014, 2015, 2016, 2017, 2018]:
    for key in [0, 1, 2]:
        if type(huffpost_data[year][key]["headline"]) != list:
            huffpost_data[year][key]["headline"] = huffpost_data[year][key][
                "headline"
            ].tolist()


# train model
for year in [2013, 2014, 2015, 2016, 2017, 2018]:
    print("Year=", year)
    m = EmbGAMClassifier(
        all_ngrams=True,
        checkpoint="bert-base-uncased",
        ngrams=2,
        random_state=42,
    )
    dset_train = huffpost_data[year][0]
    m.fit(dset_train["headline"], dset_train["category"])

    # save model
    pkl.dump(m, open(f"models/huffpost_{year}_embgam_ngrams=2.pkl", "wb"))
