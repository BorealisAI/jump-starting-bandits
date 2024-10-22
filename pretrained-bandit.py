# Copyright (c) 2024-present Royal Bank of Canada.
# All rights reserved. 
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree. 
#
import matplotlib.pyplot as plt
import random
from utils import read_generated_rewards, read_embeddings, read_arms_context, make_abs_rewards
import argparse
import time
import numpy as np

N_ARMS = 4
alpha_param = 10


def make_input_matrix(user_embeddings, users_order, n_trial, n_arms):
    available_arms = np.arange(n_arms)
    X = np.array([[user_embeddings[users_order[i]] for j in available_arms] for i in np.arange(n_trial)])
    return X


def generate_similarity_reward(arm, x):
    signal = np.dot(x, arm)
    return signal


def generate_sample_reward(reward_list):
    ans = np.mean(reward_list)
    return ans


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
        r_payoff[t] = generate_sample_reward(rewards[users_order[t]][chosen_arm])
        arm_choice[t] = chosen_arm

        A[chosen_arm] += np.outer(x_chosen_arm, x_chosen_arm.T)
        b[chosen_arm] += r_payoff[t] * x_chosen_arm

    return dict(theta=theta, p=p, arm_choice=arm_choice, r_payoff=r_payoff, A=A, b=b)


def make_regret(payoff, oracle):
    oracle = np.array(oracle)
    payoff = np.array(payoff)
    return np.cumsum(oracle - payoff)


def train(X, users_order, rewards, model=None, alpha_to_test=[alpha_param]):
    results_dict = dict()

    for alpha in alpha_to_test:
        ans = linUCB(alpha, X, users_order, rewards, model)
        results_dict[alpha] = ans
    return results_dict


def regret_testing(model, input_matrix, true_reward_matrix):
    users_order = [i for i in range(len(input_matrix))]
    np.random.shuffle(users_order)

    oracle = []
    for i in range(len(users_order)):
        max_r_t = np.max([(true_reward_matrix[users_order[i]][arm]) for arm in np.arange(N_ARMS)])
        oracle.append(max_r_t)

    print("input size:", len(input_matrix))
    X = make_input_matrix(input_matrix, users_order, n_trial=len(input_matrix), n_arms=N_ARMS)

    results_regret_pretrained = train(X, users_order, true_reward_matrix, model)
    return results_regret_pretrained[alpha_param], oracle


prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
prs.add_argument("-s", dest="samples", type=int, default=20000, required=False, help="Number of samples.\n")
prs.add_argument("-r", dest="runs", type=int, default=1, required=True, help="number of runs.\n")
prs.add_argument("-u", dest="users", type=int, default=1000, required=False, help="number of runs.\n")

args = prs.parse_args()
test_users = 1000
user_embeddings_raw = read_embeddings('./datasets/large/all_user_embeddings.csv', 64)
user_embeddings = user_embeddings_raw[0:test_users]
arms_context = read_arms_context('./datasets/arms-context.csv')

rewards_files = ['./datasets/large/pair/gpt4o-all-style-1000-rewards-percent.csv', \
                 './datasets/large/pair/gpt3.5-3-all-style-1000-rewards-percent.csv', \
                 './datasets/large/pair/anthropic3-all-style-1000-rewards-percent.csv', \
                 './datasets/large/pair/mistral-all-style-1000-rewards-percent.csv']

llm_labels = ["GPT-4o", "GPT-3.5", "Claude-3", "Mistral-small", "Similarity", "Random"]
llm_rewards = [make_abs_rewards(read_generated_rewards(rewards_files[i])) for i in range(len(rewards_files))]
similarity_reward = dict()

for lk in llm_rewards[0].keys():
    if lk >= len(user_embeddings):
        break
    similarity_reward[lk] = [[generate_similarity_reward(arm=arms_context[arm], x=user_embeddings[lk])] for arm in
                             np.arange(N_ARMS)]

similarity_reward = make_abs_rewards(similarity_reward)
llm_rewards.append(similarity_reward)

all_results = [[] for i in range(len(llm_labels))]
all_regrets = [[] for i in range(len(llm_labels))]

for q in range(args.runs):

    seed = q
    np.random.seed(seed)
    random.seed(seed)
    user_orders_0 = [i for i in range(len(user_embeddings))]
    np.random.shuffle(user_orders_0)
    users_order_1 = np.random.choice(np.arange(len(user_orders_0)), args.samples)
    users_order = []
    for i in range(len(users_order_1)):
        users_order.append(user_orders_0[users_order_1[i]])

    X = make_input_matrix(user_embeddings, users_order, n_trial=args.samples, n_arms=N_ARMS)

    llms_pretrained_dict = [train(X, users_order, llm_rewards[z]) for z in range(len(llm_rewards))]

    user_embeddings = user_embeddings_raw[0:test_users]
    not_pretrained, not_pretrained_oracle = regret_testing(None, user_embeddings, llm_rewards[0])

    for i in range(len(llms_pretrained_dict)):
        pretrained_res, pretrianed_oracle = regret_testing(llms_pretrained_dict[i][alpha_param], user_embeddings,
                                                           llm_rewards[0])
        all_results[i].append(pretrained_res['r_payoff'])
        all_regrets[i].append(make_regret(pretrained_res['r_payoff'], pretrianed_oracle))
    all_results[-1].append(not_pretrained['r_payoff'])
    all_regrets[-1].append(make_regret(not_pretrained['r_payoff'], not_pretrained_oracle))

labels = ["GPT-4o", "GPT-3.5", "Claude-3", "Mistral-small", "Similarity", "Not Pretrained"]
n_size = len(all_regrets[0][0])
x_axis = [i for i in range(n_size)]
f = plt.figure()
font = {'size': 12}
plt.rc('font', **font)
ref_reg = np.mean(all_regrets[-1], axis=0)
ref_reg = ref_reg[-1]
print(ref_reg)
for i, _ in enumerate(all_regrets):
    for j in range(len(all_regrets[i])):
        all_regrets[i][j] = all_regrets[i][j][0:n_size]
    rel_ref = -1 * np.array(all_regrets[i]).copy()
    rel_ref += all_regrets[-1]
    rel_ref /= all_regrets[-1]
    rel_ref *= 100.0
    mn_rel_ref = np.mean(rel_ref, axis=0)
    std_rel_ref = np.std(rel_ref, axis=0) / args.runs

    mn = np.mean(all_regrets[i], axis=0)
    std = 1.96 * np.std(all_regrets[i], axis=0) / np.sqrt(args.runs)
    plt.plot(x_axis, mn, label=labels[i])
    plt.fill_between(x_axis, (mn - std), (mn + std), alpha=.3)
    print("Improved Regret:", mn_rel_ref[-1], std_rel_ref[-1], labels[i])
    print(mn[-1], (ref_reg - mn[-1]) / ref_reg * 100.0, labels[i])

plt.xlabel("Number of Samples", fontsize=14)
plt.ylabel("Regret (accumulated)", fontsize=14)
plt.legend()
plt.show()
t = int(time.time())
f.savefig("synthetic-" + str(t) + ".pdf")
