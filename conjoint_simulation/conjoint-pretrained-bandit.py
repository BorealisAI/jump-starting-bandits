# Copyright (c) 2024-present Royal Bank of Canada.
# All rights reserved. 
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree. 
#

import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import argparse
import time

N_ARMS = 2

alpha_param = 10


def read_embeddings(tsvfile):
    result = []
    with open(tsvfile, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            ans = []
            for r in row:
                ans.append(int(r))
            result.append(ans)
    return result


def make_input_matrix(user_embeddings, users_order, n_trial, n_arms):
    available_arms = np.arange(n_arms)
    X = np.array([[user_embeddings[users_order[i]] for j in available_arms] for i in np.arange(n_trial)])
    return X


def generate_similarity_reward(arm, x):
    signal = np.dot(x, arm)
    return signal


def linUCB(alpha, X, users_order, rewards, pretrained_model=None):
    n_trial, n_arms, n_feature = X.shape
    print("LinUCB, alpha = ", alpha)
    arm_choice = np.empty(n_trial)
    r_payoff = np.empty(n_trial)

    theta = np.empty(shape=(n_trial, n_arms, n_feature))
    p = np.empty(shape=(n_trial, n_arms))

    A = np.array([np.diag(np.ones(shape=n_feature)) for _ in np.arange(n_arms)])
    b = np.array([np.zeros(shape=n_feature) for _ in np.arange(n_arms)])
    if pretrained_model is not None:
        A = pretrained_model['A']
        b = pretrained_model['b']

    for t in np.arange(n_trial):
        for a in np.arange(n_arms):
            inv_A = np.linalg.inv(A[a])
            theta[t, a] = inv_A.dot(b[a])
            p[t, a] = theta[t, a].dot(X[t, a]) + alpha * np.sqrt(X[t, a].dot(inv_A).dot(X[t, a]))

        chosen_arm = np.argmax(p[t])
        x_chosen_arm = X[t, chosen_arm]
        r_payoff[t] = np.mean(rewards[users_order[t]][chosen_arm])
        arm_choice[t] = chosen_arm

        A[chosen_arm] += np.outer(x_chosen_arm, x_chosen_arm.T)
        b[chosen_arm] += r_payoff[t] * x_chosen_arm

    return dict(theta=theta, p=p, arm_choice=arm_choice, r_payoff=r_payoff, A=A, b=b)


def make_regret(payoff, oracle):
    payoff = np.array(payoff)
    return np.cumsum(1 - payoff)


def train(X, users_order, rewards, model=None, alpha_to_test=[alpha_param]):
    results_dict = dict()
    for alpha in alpha_to_test:
        ans = linUCB(alpha, X, users_order, rewards, model)
        results_dict[alpha] = ans
    return results_dict


def read_data(file):
    ans = []
    with open(file, 'r', newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            t = []
            for r in row:
                t.append(r.strip())
            ans.append(t)
    return ans


def parse_generated_responses(responses):
    generated_responses = []
    for i in range(len(responses)):
        user_id = int(responses[i][0])
        raw_ans = responses[i][3]

        split_ans = raw_ans.split(".")
        ans = -1
        if "Vaccine A" in split_ans[-1]:
            ans = 0
        elif "Vaccine B" in split_ans[-1]:
            ans = 1
        elif "Vaccine A" in split_ans[-2]:
            ans = 0
        elif "Vaccine B" in split_ans[-2]:
            ans = 1
        else:
            if "I would choose Vaccine A" or "I will choose Vaccine A" in raw_ans:
                ans = 0
            elif "I would choose Vaccine B" or "I will choose Vaccine B" in raw_ans:
                ans = 1
            else:
                print(user_id, raw_ans)
        t = [0, 0]
        t[ans] = 1
        generated_responses.append(t)
    return np.array(generated_responses)


def generate_true_responses(vaccines):
    true_responses = dict()
    for i in range(0, len(vaccines), 2):
        vac = vaccines[i]
        user_id = int(vac[0])
        resp = [0, 0]
        if int(vaccines[i][2]) == 1:
            resp[0] += 1
        if int(vaccines[i + 1][2]) == 1:
            resp[1] += 1
        if user_id not in true_responses:
            true_responses[user_id] = []
        if resp[0] == resp[1] and resp[0] == 0:
            resp = [1, 1]
        true_responses[user_id].append(resp)
    return true_responses


def read_and_train(input_type, samples):
    print(input_type)
    N_TRIAL = samples
    vaccine_embeddings = read_embeddings("../surveys/conjoint_covid/cf_vaccines_embeddings.tsv")

    if input_type == "personal":
        raw_user_embeddings = read_embeddings("../surveys/conjoint_covid/no-personal-10k-embedding.tsv")
        conjoint_data = read_data("../surveys/conjoint_covid/no-personal-10k.tsv")
        raw_rewards = read_data("../surveys/conjoint_covid/3.5-no-personal-cf_preferences_10k-total.tsv")

    elif input_type == "vaccine":
        raw_user_embeddings = read_embeddings("../surveys/conjoint_covid/no-vaccines-10k-embedding.tsv")
        conjoint_data = read_data("../surveys/conjoint_covid/no-vaccines-10k.tsv")
        raw_rewards = read_data("../surveys/conjoint_covid/3.5-no-vaccines-cf_preferences_10k-total.tsv")

    elif input_type == "partial":
        raw_user_embeddings = read_embeddings("../surveys/conjoint_covid/no-vac-partial-10k-embedding.tsv")
        conjoint_data = read_data("../surveys/conjoint_covid/no-vac-partial-10k.tsv")
        raw_rewards = read_data("../surveys/conjoint_covid/3.5-partial-pereferences_10k-total.tsv")

    elif input_type == "gpt4":
        raw_user_embeddings = read_embeddings("../surveys/conjoint_covid/user-gpt4-embedding.tsv")
        conjoint_data = read_data("../surveys/conjoint_covid/user-gpt4.tsv")
        raw_rewards = read_data("../surveys/conjoint_covid/gpt4o-preferences_10k-total.tsv")

    elif input_type == "true":
        raw_user_embeddings = read_embeddings("../surveys/conjoint_covid/actual_users_embedding.tsv")
        conjoint_data = read_data("../surveys/conjoint_covid/cf_true_users_total.tsv")
        raw_rewards = read_data("../surveys/conjoint_covid/cf_true_users_total.tsv")
        vaccine_embeddings = read_embeddings("../surveys/conjoint_covid/actual_vaccine_embedding.tsv")

    elif input_type == "anthropic":
        raw_user_embeddings = read_embeddings("../surveys/conjoint_covid/user-anthropic-embedding.tsv")
        conjoint_data = read_data("../surveys/conjoint_covid/user-anthropic.tsv")
        raw_rewards = read_data("../surveys/conjoint_covid/anthropic-haiku-preferences_10k-total.tsv")

    elif input_type == "mistral":
        raw_user_embeddings = read_embeddings("../surveys/conjoint_covid/user-mistral-embedding.tsv")
        conjoint_data = read_data("../surveys/conjoint_covid/user-mistral.tsv")
        raw_rewards = read_data("../surveys/conjoint_covid/mistral-preferences_10k-total.tsv")

    else:
        raw_user_embeddings = read_embeddings("../surveys/conjoint_covid/cf_user_embedding_20k.tsv")
        conjoint_data = read_data("../surveys/conjoint_covid/cf_users_20k.tsv")
        raw_rewards = read_data("../surveys/conjoint_covid/3.5-cf_preferences_20k_0.tsv")

    conjoint_data = conjoint_data[0:samples]
    raw_user_embeddings = raw_user_embeddings[0:samples]
    raw_rewards = raw_rewards[0:samples]

    user_embeddings = []
    if input_type != "true":
        for i in range(len(conjoint_data)):
            vacc_a = int(conjoint_data[i][0])
            vacc_b = int(conjoint_data[i][1])
            raw_user_vec = raw_user_embeddings[i].copy()
            vec_a = np.array(vaccine_embeddings[vacc_a])
            vec_b = np.array(vaccine_embeddings[vacc_b])
            vec_c = np.outer(raw_user_vec, vec_a - vec_b)
            c_flat = vec_c.flatten()
            emb = c_flat.copy()

            user_embeddings.append(emb)
    else:

        for i in range(len(conjoint_data)):
            vacc_a = i * 2
            vacc_b = i * 2 + 1
            emb = raw_user_embeddings[int(conjoint_data[i][0])].copy()
            emb.extend(vaccine_embeddings[vacc_a])
            emb.extend(vaccine_embeddings[vacc_b])
            user_embeddings.append(emb)

    users_order = [i for i in range(len(user_embeddings))]
    users_order = 2 * users_order

    np.random.shuffle(users_order)
    if N_TRIAL < len(user_embeddings):
        users_order = users_order[0:N_TRIAL]
    print("input size:", len(user_embeddings), len(users_order))
    X = make_input_matrix(user_embeddings, users_order, n_trial=N_TRIAL, n_arms=N_ARMS)

    rewards = parse_generated_responses(raw_rewards)

    payoff_random = []
    for i in range(N_TRIAL):
        random_arm = np.random.choice(N_ARMS)
        r_t = rewards[users_order[i]][random_arm]
        payoff_random.append(r_t)

    payoff_random = np.array(payoff_random)

    results_regret = train(X, users_order, rewards)
    result_dict = dict()
    for alpha in results_regret.keys():
        k = str(alpha)
        result_dict[k] = results_regret[alpha]

    result_dict['random'] = {'r_payoff': payoff_random}

    return results_regret


def regret_testing(args):
    model = read_and_train(args.type, args.samples)
    actual_vaccines = read_data("../surveys/conjoint_covid/vaccine_info.tsv")
    raw_true_user_embedding = read_embeddings("../surveys/conjoint_covid/actual_users_embedding.tsv")
    true_vaccine_embedding = read_embeddings("../surveys/conjoint_covid/actual_vaccine_embedding.tsv")
    true_resposnes = generate_true_responses(actual_vaccines)
    delete_list = []
    if args.type == "personal":
        delete_list = [i for i in range(1, 10)]

    if args.type == "vaccine":
        delete_list = [i for i in range(10, 18)]

    if args.type == "partial":
        delete_list = [i for i in range(5, 18)]

    input_matrix = []
    true_reward_matrix = []
    for i in range(0, len(actual_vaccines), 10):
        exp_id = 0
        user_id = int(actual_vaccines[i][0])
        for j in range(i, i + 10, 2):
            user_t = raw_true_user_embedding[user_id - 1]
            if len(delete_list) > 0:
                user_t = np.delete(user_t, delete_list)
                user_t = list(user_t)
            vec_a = np.array(true_vaccine_embedding[j])
            vec_b = np.array(true_vaccine_embedding[j + 1])
            raw_user_vec = user_t
            vec_c = np.outer(raw_user_vec, vec_a - vec_b)
            c_flat = vec_c.flatten()
            emb = c_flat.copy()
            input_matrix.append(emb)
            true_reward_matrix.append(true_resposnes[user_id][exp_id])
            exp_id += 1
    input_matrix = np.array(input_matrix)
    true_reward_matrix = np.array(true_reward_matrix)

    users_order = [i for i in range(len(input_matrix))]
    np.random.shuffle(users_order)

    print("input size:", len(input_matrix))
    X = make_input_matrix(input_matrix, users_order, n_trial=len(input_matrix), n_arms=N_ARMS)

    results_regret_pretrained = train(X, users_order, true_reward_matrix, model[alpha_param])
    result_regret_zero = train(X, users_order, true_reward_matrix)
    result_dict = {"pretrained": results_regret_pretrained[alpha_param],
                   "not pretrained": result_regret_zero[alpha_param]}
    return result_dict


if __name__ == "__main__":

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    prs.add_argument("-t", dest="type", type=str, default="gpt3.5", required=False, help="Type of data.\n")
    prs.add_argument("-s", dest="samples", type=int, default=10000, required=False, help="Number of samples.\n")
    prs.add_argument("-r", dest="runs", type=int, default=1, required=True, help="number of runs.\n")

    args = prs.parse_args()

    types = ["gpt3.5", "anthropic", "gpt4", "mistral"]

    randoms = []
    results = [[] for _ in range(len(types))]
    for i in range(1, args.runs + 1):
        print("----------------------", i, "----------------------")
        seed = i
        np.random.seed(seed)
        random.seed(seed)
        for j in range(len(types)):
            args_t = args
            args.type = types[j]
            result_dict = regret_testing(args_t)
            results[j].append(result_dict['pretrained']['r_payoff'])
            if j == 1:
                randoms.append(result_dict['not pretrained']['r_payoff'])

    f = plt.figure()
    font = {'size': 12}
    plt.rc('font', **font)
    regrets_pretrained = [[make_regret(payoff=x, oracle=None) for x in results[i]] for i in range(len(results))]
    regrets_nonpretrained = [make_regret(payoff=x, oracle=None) for x in randoms]
    all_regrets = regrets_pretrained
    all_regrets.append(regrets_nonpretrained)
    labels = ["GPT-3.5", "Claude-3", "GPT-4o", "Mistral-small", "Not Pretrained"]
    n_size = len(all_regrets[0][0])

    x_axis = [i for i in range(n_size)]

    ref_reg = np.mean(all_regrets[-1], axis=0)

    csv_file = "./regrets/regrets"
    for l in labels:
        csv_file = csv_file + "_" + l
    t = int(time.time())
    csv_file += str(args.samples) + "---" + str(t) + ".csv"
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        for i in range(len(all_regrets)):
            for j in range(len(all_regrets[i])):
                writer.writerow([labels[i], j, all_regrets[i][j].tolist()])

    for i, _ in enumerate(all_regrets):
        for j in range(len(all_regrets[i])):
            all_regrets[i][j] = all_regrets[i][j][0:n_size]
        mn = np.mean(all_regrets[i], axis=0)
        std = 1.96 * np.std(all_regrets[i], axis=0) / np.sqrt(args.runs)
        plt.plot(x_axis[0:1000], mn[0:1000], label=labels[i])
        plt.fill_between(x_axis[0:1000], (mn - std)[0:1000], (mn + std)[0:1000], alpha=.3)

        rel_ref = -1 * np.array(all_regrets[i]).copy()
        rel_ref += all_regrets[-1]
        rel_ref /= all_regrets[-1]
        rel_ref *= 100.0
        mn_rel_ref = np.mean(rel_ref, axis=0)
        std_rel_ref = np.std(rel_ref, axis=0) / args.runs
        print(1000, "##", mn_rel_ref[999], std_rel_ref[999], labels[i])
        print(2000, "##", mn_rel_ref[1999], std_rel_ref[1999], labels[i])
        print(5000, "##", mn_rel_ref[4999], std_rel_ref[4999], labels[i])
        print(-1, "##", mn_rel_ref[-1], std_rel_ref[-1], labels[i])

    plt.xlabel("Number of Samples", fontsize=14)
    plt.ylabel("Regret (accumulated)", fontsize=14)
    plt.legend()
    plt.show()
    t = int(time.time())
    f.savefig("./plots/pretrained-vs-zero" + str(t) + ".pdf")
