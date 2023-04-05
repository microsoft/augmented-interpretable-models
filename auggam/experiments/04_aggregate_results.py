import embgam.data as data
import experiments.config as config
from os.path import join as oj

if __name__ == '__main__':
    # rs_vary_ngrams_test = data.load_fitted_results(
        # fname_filters=['ngtest'],
        # results_dir_main=config.results_dir,
    # )
    # rs_vary_ngrams_test.to_pickle(
        # oj(config.results_dir, 'rs_vary_ngrams_test.pkl'))

    rs = data.load_fitted_results(results_dir_main=config.results_dir)
    rs.to_pickle(oj(config.results_dir, 'fitted_results_aggregated.pkl'))
