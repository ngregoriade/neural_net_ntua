# first line: 392
def _fit_resample_one(sampler, X, y, message_clsname="", message=None, **fit_params):
    with _print_elapsed_time(message_clsname, message):
        X_res, y_res = sampler.fit_resample(X, y, **fit_params)

        return X_res, y_res, sampler
