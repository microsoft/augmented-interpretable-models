import random


def clean_headlines(huffpost_data):
    """
    consolidate all the text to be the same data type (str instead of np.str_)
    and convert all arrays to lists

    Args:
        huffpost_data (dict<int: dict>): dict mapping year to data
    Returns:
        huffpost_data (with values modified)
    """
    # the keys 0, 1, 2 represent indistribution (ID) training and validation and out of distribution test set
    # see table 6 for more info: https://arxiv.org/pdf/2211.14238.pdf
    for year in [2012, 2013, 2014, 2015, 2016, 2017, 2018]:
        for key in [0, 1, 2]:
            if type(huffpost_data[year][key]["headline"]) != list:
                huffpost_data[year][key]["headline"] = huffpost_data[year][key][
                    "headline"
                ].tolist()
                huffpost_data[year][key]["category"] = huffpost_data[year][key][
                    "category"
                ].tolist()

    return huffpost_data


def sample_data(huffpost_data, year, in_dist=True, frac=1):
    """
    sample a fraction of data from a given year from the huffpost dataset

    Args:
        huffpost_data (dict<int: dict>): dict mapping year to data
        year (int): integer from 2012-2018
        in_dist (bool): if true, sample from the in distribution data, o.w OOD
        frac (float): fraction of points to sample from the year
    Returns:
        data (list<str>): list of headlines
        labels (list<int>): list of labels
    """

    assert frac >= 0 and frac <= 1, "Not a valid fraction"
    assert any(
        [year == y for y in [2012, 2013, 2014, 2015, 2016, 2017, 2018]]
    ), "Not a valid year"
    if in_dist:
        data = huffpost_data[year][0]["headline"] + huffpost_data[year][1]["headline"]
        labels = huffpost_data[year][0]["category"] + huffpost_data[year][1]["category"]
    else:
        data = huffpost_data[year][2]["headline"]
        labels = huffpost_data[year][2]["category"]

    if frac == 1:
        return data, labels

    assert len(data) == len(labels), "Number of data points and labels don't match!"
    inds = set(random.sample(list(range(len(data))), int(frac * len(data))))

    return [data[i] for i in inds], [labels[i] for i in inds]
