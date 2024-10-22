# Copyright (c) 2024-present Royal Bank of Canada.
# All rights reserved. 
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree. 
#

import csv


arm_names = ['Formal Approach', 'Emotional/Narrative Style', 'Informative/Educational Style',
             'Personal/Relatable Style']
arm_num = 4


def read_arms(csv_file):
    arms = dict()
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            user_id = int(row[0])
            if user_id >= 1000:
                break
            arm_id = row[1]
            arm_content = row[2]
            for k in range(len(arm_names)):
                if arm_names[k] == arm_id:
                    break
            if user_id not in arms:
                arms[user_id] = []
            arms[user_id].append(arm_content)

    return arms


def write_user_profiles(user_profiles, csv_file):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for profile in user_profiles:
            details = profile.split('\n')
            for i in range(len(details)):
                details[i] = details[i].strip()

            writer.writerow(details)


def read_embeddings(csv_file, embedding_size=64):
    embeddings = []
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            t = []
            for i in range(1, embedding_size + 1):
                t.append(float(row[i]))
            embeddings.append(t)
    return embeddings


def read_arms_context(arms_file, arms_num=arm_num, embedding_size=64):
    arms_context = []
    with open(arms_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i < arms_num:
                continue
            t = []
            for j in range(embedding_size):
                t.append(float(row[j]))
            arms_context.append(t)
    return arms_context


def read_generated_rewards(reward_file='./datasets/rewards.csv'):
    rewards = dict()

    with open(reward_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            user = int(row[0])
            arm = int(row[1])

            for j in range(2, len(row)):
                s = row[j].strip()
                s = s.replace('\\n', '')
                score = float(s)
                if user not in rewards:
                    rewards[user] = [[] for _ in range(arm_num)]
                rewards[user][arm].append(score)
    return rewards


def make_abs_rewards(in_rewards):
    out_rewards = dict()
    for key in in_rewards.keys():
        out_rewards[key] = []
        tmp_rw = []
        t = 0
        for arm in range(len(in_rewards[key])):
            t += in_rewards[key][arm][0]
        for arm in range(len(in_rewards[key])):
            tmp_rw.append(in_rewards[key][arm][0] * 1.0 / t)
        out_rewards[key] = tmp_rw
    return out_rewards
