# Copyright (c) 2024-present Royal Bank of Canada.
# All rights reserved. 
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree. 
#

import csv
import itertools
import numpy as np
import openai
import time
import anthropic
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import argparse

ti = int(time.time())
np.random.seed(ti)

raw_file_path = "../surveys/conjoint_covid/Kreps_etal_vax_replication_data.tab"
all_data = []
column_path = "../surveys/conjoint_covid/columns.csv"
columns = dict()
states = []

with open(column_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
        columns[row[0].strip()] = i

with open('../surveys/conjoint_covid/states.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        states.append(row[0].strip())

with open(raw_file_path, 'r', newline='') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        t = []
        for r in row:
            t.append(r.strip())
        all_data.append(t)


def get_chat_completion(prompt, model, temp=0.5):
    if model == "gpt3.5":
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},

            ],
            temperature=temp)
        return response.choices[0].message.content
    elif model == "gpt4":
        response = openai.ChatCompletion.create(
            model="gpt-4-o",
            messages=[
                {"role": "user", "content": prompt},

            ],
            temperature=temp)
        return response.choices[0].message.content
    elif model == "claude":
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            temperature=0.0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    elif model == "mistral":
        model = "mistral-small-2402"
        client = MistralClient()
        chat_response = client.chat(
            model=model,
            messages=[ChatMessage(role="user", content=prompt, temperature=temp)]
        )

        return chat_response.choices[0].message.content


def generate_counterfactual_vaccines(data):
    vaccine_values = [dict() for _ in range(7)]
    unique_vaccines = dict()
    for user in data:
        id = 0
        unique_features = ""
        for i in range(columns['efficacy'], columns['endorsed'] + 1):
            unique_features += user[i]
            if user[i] not in vaccine_values[id]:
                vaccine_values[id][user[i]] = 1
            id += 1
        if unique_features not in unique_vaccines:
            unique_vaccines[unique_features] = 0
        unique_vaccines[unique_features] += 1
    vaccine_values_list = [vaccine_values[i].keys() for i in range(len(vaccine_values))]
    cf_vacc = []

    for element in itertools.product(*vaccine_values_list):
        cf_vacc.append(list(element))

    with open("../surveys/conjoint_covid/cf_vaccines.tsv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for vacc in cf_vacc:
            writer.writerow(vacc)
    return vaccine_values


def generate_counterfactual_users(filename, sample_size=10000, no_personal=False, no_vaccine=False,
                                  no_vac_partial_pers=False):
    delete_list = []
    if no_personal:
        delete_list = [i for i in range(1, 10)]

    if no_vaccine:
        delete_list = [i for i in range(10, 18)]

    if no_vac_partial_pers:
        delete_list = [i for i in range(5, 18)]

    race = ["white", "black", "latino"]
    salary = ["below $20,000. ", "between $20,000 and $40,000. ", "between $40,000 and $60,000. ",
              "between $60,000 and $80,000. ", "between $80,000 and $100,000. ", "more than $100,000. "]
    relig = ["Protestant", "Roman Catholic", "Mormon", "Orthodox such as Greek or Russian Orthodox", "Jewish", "Muslim",
             "Buddhist", "Hindu", "Atheist", "Agnostic", "nothing in particular"]
    edu = ["less than High School", "High School / GED", "Some College", "2-year College Degree",
           "4-year College Degree", "Master's Degree", "Doctoral Degree", "Professional Degree"]

    ideology = ["Liberal", "moderate", "Conservative"]
    political = ["Republican", "Democrat", "Independent", "undecided"]
    pres_approval = ["approves of the way Donald Trump is handling his job as president. ",
                     "disapproves of the way Donald Trump is handling his job as president. "]
    work_home = ["can work fully from home. ", "can't work from home.  ", "can work from home to some extent.  "]
    personal_contact1 = ["knows someone who has been officially tested positive for Covid. ",
                         "doesn't know someone who has been officially tested positive for Covid. "]
    personal_contact2 = ["knows someone who has been hospitalized or died as a result of having COVID. ",
                         "doesn't know someone who has been hospitalized or died as a result of having COVID. "]
    worst = ["believes that in the covid outbreak the worst is yet to come. ", ""]
    china_hid = ["believes that China hid the coronavirus from the rest of the world after its outbreak in Wuhan. ", ""]
    flu = ["never got flu vaccine in the past. ", "got flu vaccine one or twice in the past. ",
           "got flu vaccine most years in the past. "]
    mandatory = [
        "thinks that all children should be required to be vaccinated against childhood diseases, such as measles, mumps, rubella, and polio. ",
        "thinks that parents should be able to decide NOT to vaccinate their children against childhood diseases, such as measles, mumps, rubella, and polio. "]
    vacc_safe = ["safe", "somewhat safe", "not at all safe"]
    all_vars = [race, states, salary, edu, relig, ideology, political, pres_approval, work_home, personal_contact1,
                personal_contact2, worst, china_hid, flu, mandatory, vacc_safe]
    sample_users = []
    sample_users_embedding = []
    for x in range(sample_size):
        user_embedding = []
        text = "This user is a "
        t = np.random.randint(2)
        pronoun = "She "
        obj_pronoun = "Her "
        low_obj = "her "
        sex_sample = " woman. "
        user_embedding.append(0)
        if t == 1:
            pronoun = "He "
            obj_pronoun = "His "
            low_obj = "his "
            sex_sample = " man. "
            user_embedding[-1] = 1
        age = np.random.randint(20, 60)
        user_embedding.append(age)
        text += str(age) + " year-old "
        z = np.random.choice(race)
        user_embedding.append(z)
        text += z + sex_sample
        z = np.random.choice(states)
        user_embedding.append(z)
        text += pronoun + "lives in " + z + ". "
        z = np.random.choice(salary)
        user_embedding.append(z)
        text += obj_pronoun + "annual salary of is " + z
        z = np.random.choice(edu)
        user_embedding.append(z)

        no_vac_partial_pers = text

        text += obj_pronoun + "highest education is " + z + ". "
        z = np.random.choice(relig)
        user_embedding.append(z)
        text += pronoun + "describes " + low_obj + "religious beliefs as " + z + ". "
        z = np.random.choice(ideology)
        user_embedding.append(z)
        text += pronoun + "considers " + low_obj + "self as " + z + ". "
        z = np.random.choice(political)
        user_embedding.append(z)
        text += pronoun + "considers " + low_obj + "political views as " + z + ". "
        z = np.random.choice(pres_approval)
        user_embedding.append(z)
        text += pronoun + z

        if no_personal:
            text = "This user is a" + sex_sample

        no_vaccine_text = text

        z = np.random.choice(work_home)
        user_embedding.append(z)
        text += pronoun + z
        t = np.random.randint(2)
        z = t
        user_embedding.append(z)
        if t == 0:
            text += pronoun + "knows someone who has been officially tested positive for Covid. "
            z = np.random.choice(personal_contact2)
            user_embedding.append(z)
            text += pronoun + z

        else:
            text += pronoun + "doesn't know someone who has been officially tested positive for Covid. "
            text += pronoun + "doesn't know someone who has been hospitalized or died as a result of having COVID. "
            user_embedding.append(0)
        t = np.random.randint(2)
        z = t
        user_embedding.append(z)
        if t == 0:
            text += pronoun + "believes that in the covid outbreak the worst is yet to come. "
        t = np.random.randint(2)
        z = t
        user_embedding.append(z)
        if t == 0:
            text += pronoun + "believes that China hid the coronavirus from the rest of the world after its outbreak in Wuhan. "
        z = np.random.choice(flu)
        user_embedding.append(z)
        text += pronoun + z
        z = np.random.choice(mandatory)
        user_embedding.append(z)
        text += pronoun + z
        z = np.random.choice(vacc_safe)
        user_embedding.append(z)
        text += pronoun + "thinks in general vaccines are " + z + ". "
        pairs = dict()
        if no_vaccine:
            text = no_vaccine_text
        if no_vac_partial_pers:
            text = no_vac_partial_pers
        for j in range(1):
            vaccine_A = np.random.randint(576)
            vaccine_B = np.random.randint(576)
            while (vaccine_B == vaccine_A):
                vaccine_B = np.random.randint(576)
            while (vaccine_A, vaccine_B) in pairs:
                vaccine_A = np.random.randint(576)
                vaccine_B = np.random.randint(576)
                while (vaccine_B == vaccine_A):
                    vaccine_B = np.random.randint(576)
            pairs[(vaccine_A, vaccine_B)] = 1
            pairs[(vaccine_B, vaccine_A)] = 1

            sample_users.append([vaccine_A, vaccine_B, text])
            sample_users_embedding.append(user_embedding)

    full_name = "../surveys/conjoint_covid/" + filename + ".tsv"
    full_embedding = "../surveys/conjoint_covid/" + filename + "-embedding.tsv"
    with open(full_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for s_user in sample_users:
            writer.writerow(s_user)

    with open(full_embedding, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for s_user in sample_users_embedding:
            binary_embedding = [s_user[0], s_user[1]]
            if len(delete_list) > 0 and delete_list[0] == 1:
                binary_embedding = [s_user[0]]
            for i in range(2, len(s_user)):
                if i in delete_list:
                    continue
                if s_user[i] == 0 or s_user[i] == 1:
                    binary_embedding.append(s_user[i])
                else:
                    ind = all_vars[i - 2].index(s_user[i])
                    binary_embedding.append(ind)
            writer.writerow(binary_embedding)


def read_data(r_files):
    result = []
    with open(r_files, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            t = []
            for r in row:
                t.append(r)
            result.append(t)
    return result


def generate_vaccine_embeddings(vaccines):
    embedding_values = []
    for i in range(len(vaccines)):
        t = [j for j in range(len(vaccines[i].keys()))]
        embedding_values.append(t)

    vacc_embeddings = []
    for element in itertools.product(*embedding_values):
        vacc_embeddings.append(list(element))

    with open("../surveys/conjoint_covid/cf_vaccines_embeddings.tsv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for vacc in vacc_embeddings:
            writer.writerow(vacc)


def generate_vaccine_text(vaccine_id, vac_data):
    text = "Vaccine " + vaccine_id + " is described as follows: The efficacy of it is " + vac_data[0]
    text += ", and the protection duration is " + vac_data[1]
    text += ". Vaccine " + vaccine_id + " shows major side effects " + vac_data[2] + " and shows minor side effects " + \
            vac_data[3]
    text += ". " + vac_data[4] + ", and it is originated in " + vac_data[5] + ". "

    text += "Vaccine " + vaccine_id + " is also endorsed by " + vac_data[6] + ". "

    return text


def writeinfo(w_info, w_file):
    with open(w_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for v in w_info:
            writer.writerow(v)


def generate_conjoint_responses(num, llm):
    users = read_data("../surveys/conjoint_covid/cf_users.tsv")
    vaccines = read_data("../surveys/conjoint_covid/cf_vaccines.tsv")

    results = []
    length = 1000
    st = num * length
    ed = (num + 1) * length
    for i in range(st, ed):
        print(i, st, ed)
        A = int(users[i][0])
        B = int(users[i][1])
        vac_A = vaccines[A]
        vac_B = vaccines[B]
        user_id = i

        text = "Consider you are in the middle of the covid pandemic where vaccines are just being produced. Pretend to be the following user: [Start of User] "
        text += users[i][2]
        text += "[End of user] now you are given two vaccine choices for covid. The description of each vaccine is as follows: [Start of Vaccine A] "
        text += generate_vaccine_text('A', vac_A)
        text += "[End of Vaccine A] Now the next one: [Start of Vaccine B] " + generate_vaccine_text('B',
                                                                                                     vac_B) + " [End Vaccine B] "
        text += "Which one do you take? A or B? Let's think step by step. Print the final answer as [Final Answer] at the end as well."

        response = get_chat_completion(text, llm)
        response = response.replace("\n", " ")
        results.append([user_id, A, B, response])
        if i % 50 == 0 and len(results) > 0:
            writeinfo(results, "../surveys/conjoint_covid/3.5-partial-pereferences_10k" + str(num) + ".tsv")
            results = []

    if len(results) > 0:
        writeinfo(results, "../surveys/conjoint_covid/3.5-partial-pereferences_10k" + str(num) + ".tsv")


prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
prs.add_argument("-p", dest="prefix", type=str, default="./datasets/large/", required=False, help="Name prefix.\n")
prs.add_argument("-n", dest="num", type=int, default=0, required=False, help="Run number.\n")
prs.add_argument("-m", dest="model", type=str, default="gpt3.5", required=False, help="LLM.\n")

args = prs.parse_args()

generate_counterfactual_users("cf_users", 10000, False, False)
unique_vaccines = generate_counterfactual_vaccines(all_data)
generate_vaccine_embeddings(unique_vaccines)
generate_conjoint_responses(args.num, args.model)
