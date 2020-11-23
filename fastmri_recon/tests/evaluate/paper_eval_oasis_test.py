from fastmri_recon.evaluate.scripts.paper_eval_oasis import evaluate_paper_oasis


def test_evaluate_paper_oasis():
    # TODO: add a test on the metrics values
    evaluate_paper_oasis(n_samples=2)
