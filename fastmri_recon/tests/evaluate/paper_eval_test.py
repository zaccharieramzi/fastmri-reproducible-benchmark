from fastmri_recon.evaluate.scripts.paper_eval import evaluate_paper


def test_evaluate_paper():
    # TODO: add a test on the metrics values
    evaluate_paper(n_samples=2)
