def BivariateHawkes_LL(params_array, events_list, end_time, M=2):
    """
    Calculate multivariate Hawkes process log-likelihood(M=2 for bivariate)

    :param params_array: contains all parameters needed for calculating Hawkes process
    :param events_lists: contains two list of timestamps
    :param end_time: end duration time
    :param M: M = 2 indicate bivariate Hawkes process

    :return: log-likelihood of the multivariate Hawkes process
    """ 
    
    
    mu_array, alpha_array, beta, q = params_array
    
    alpha_array = np.resize(alpha_array, [2,2])
    # first term
    first = - np.sum(mu_array)*end_time
    # second term
    second = 0
    for m in range(M): 
        for v in range(M):
            for b in range(len(q)):
                if len(events_list[v]) == 0:
                    continue
                second -= alpha_array[m, v] * q[b] * np.sum(1 - np.exp(-beta[b] * (end_time - events_list[v])))
    # third term
    third = 0
    for m in range(M):
        for k in range(len(events_list[m])):
            tmk = events_list[m][k]
            inter_sum = 0
            for v in range(M):
                for b in range(len(q)):
                    if len(events_list[v]) == 0:
                        continue
                    v_less = events_list[v][events_list[v] < tmk]
                    Rmvk = np.sum(np.exp(-beta[b] * (tmk - v_less)))
                    inter_sum += alpha_array[m, v] * beta[b] * q[b] * Rmvk
            if mu_array[m] + inter_sum == 0:
                third += 0
            else: 
                third += np.log(mu_array[m] + inter_sum)
    return (first+second+third)
