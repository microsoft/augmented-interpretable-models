# Should unzip the babylm data here


# BabyLM Dataset
This download includes LM Pretraining data for the 2023 CoNLL/CMCL shared task, [The BabyLM Challenge](https://babylm.github.io/). The (unzipped) data is not large, only ~700MB.

## Contents of this download
- `babylm_100M`: 100M-word training set for the *strict* track.
- `babylm_10M`: 10M-word training set for the *strict-small* track.
- `babylm_dev`: Development set for both tracks (10M words)
- `babylm_test`: Test set for both tracks (10M words)

Each directory above contains a single `.txt` file from each of the 10 domains listed below.

## Composition of the data
All datasets are sampled from a mixture of 10 data domains, shown below, along with their respective weights in the distributed dataset.

| Source | Weight | Domain | Citation | Website | License |
| --- | --- | --- | --- | --- | --- |
| OpenSubtitles | 30% | Dialogue, Scripted | Lison & Tiedermann (2016) | [link](https://opus.nlpl.eu/OpenSubtitles-v2018.php) | Open source |
| Simple English Wikipedia | 15% | Nonfiction | -- | [link](https://dumps.wikimedia.org/simplewiki/20221201/) | [link](https://dumps.wikimedia.org/legal.html) |
| BNC | 10% | Dialogue | BNC Consortium (2007) | [link](http://www.natcorp.ox.ac.uk/) | [link](http://www.natcorp.ox.ac.uk/docs/licence.html) <sup>1</sup> |
| Project Gutenberg | 10% | Fiction, Nonfiction | Gerlach & Font-Clos (2020) | [link](https://github.com/pgcorpus/gutenberg) | [link](https://www.gutenberg.org/policy/license.html) |
| QED | 10% | Dialogue, Education | Abdelali et al. (2014) | [link](https://opus.nlpl.eu/QED.php) | [link](https://opus.nlpl.eu/QED.php) |
| Wikipedia | 10% | Nonfiction | -- | [link](https://dumps.wikimedia.org/enwiki/20221220/) | [link](https://dumps.wikimedia.org/legal.html) |
| Children's Book Test | 6% | Fiction, Child-Directed | Hill et al. (2016) | [link](https://research.facebook.com/downloads/babi/) | Public domain |
| CHILDES | 4% | Dialogue, Child-Directed | MacWhinney (2000) | | [link](https://talkbank.org/share/rules.html) |
| Children's Stories | 4% | Fiction, Child-Directed | -- | [link](https://www.kaggle.com/datasets/edenbd/children-stories-text-corpus) | Public domain |
| Switchboard | 1% | Dialogue | Godfrey et al. (1992), Stolcke et al., (2000) | [link](http://compprag.christopherpotts.net/swda.html) | [link](http://compprag.christopherpotts.net/swda.html) |

<sup>1</sup> Our distribution of part of the BNC Texts is permitted under the fair dealings provision of copyright law (see term (2g) in the BNC license).


## Data preprocessing

Data was minimally preprocessed to conform to a plain text format. We did not tokenize the data. Documents are not necessarily complete are newline separated.

For documentation of the preprocessing pipeline, consult the following repo: https://github.com/babylm/babylm_data_preprocessing


## References
Abdelali, A., Guzman, F., Sajjad, H., & Vogel, S. (2014). The AMARA Corpus: Building parallel language resources for the educational domain. In Proceedings of the 9th International Conference on Language Resources and Evaluation (LREC 2014). 1856-1862.

BNC Consortium. (2007). The British National Corpus, XML Edition. Oxford Text Archive, http://hdl.handle.net/20.500.12024/2554.

Gerlach, M., & Font-Clos, F. (2020). A standardized Project Gutenberg corpus for statistical analysis of natural language and quantitative linguistics. Entropy, 22(1), 126.

Godfrey, J. J., Holliman, E. C., & McDaniel, J. (1992). SWITCHBOARD: Telephone speech corpus for research and development. In Acoustics, Speech, and Signal Processing, IEEE International Conference on (Vol. 1, pp. 517-520). IEEE Computer Society.

Hill, F., Bordes, A., Chopra, S., Weston, J. (2016). The Goldilocks principle: Reading children’s books with explicit memory representations. In Proceedings of the 4th International Conference on Learning Representations (ICLR 2016).

Lison, P. & Tiedemann, J. (2016). OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016).

MacWhinney, B. (2000). The CHILDES Project: Tools for analyzing talk. Third Edition. Mahwah, NJ: Lawrence Erlbaum Associates.

Stolcke, A., Ries, K., Coccaro, N., Shriberg, E., Bates, R., Jurafsky, D., Taylor, P., Martin, R., Van Ess-Dykema, C., & Meteer, M. (2000). Dialogue act modeling for automatic tagging and recognition of conversational speech. Computational linguistics, 26(3), 339-373.

Tiedemann, J. (2012). Parallel Data, Tools and Interfaces in OPUS. In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012).