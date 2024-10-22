# Copyright (c) 2024-present Royal Bank of Canada.
# All rights reserved. 
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree. 
#


import openai
import argparse
from utils import write_user_profiles
import anthropic
import csv

embedding_size = 64

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


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


styles = [
    "Formal Approach: Begin with a formal greeting, introduce the organization, highlight the current need, explain how their contribution can make a difference, and end with a polite request asking for their generous support. The tone is official and respectful, focusing on the importance of the cause.",
    "Emotional/Narrative Style: This style leverages storytelling to evoke empathy and compassion from the reader. You can share a real-life story related to the cause, emphasize the struggle, and showcase how their donation can change lives.",
    "Informative/Educational Style: This style relies on facts, statistics, and evidence to persuade the user to donate. It educates the reader about the cause, its impact, and how the charity is fighting for it. The reader's decision will be driven by the evidence of how efficient the charity work is.",
    "Personal/Relatable Style: Here, you use a more casual, friendly tone. You could even share personal experiences with the charity or testimonies from donors. The essence is to make the reader feel closely connected and to understand that anyone can make a difference."]
style_names = ["Formal Approach", "Emotional/Narrative Style", "Informative/Educational Style",
               "Personal/Relatable Style"]


def separate_profiles(profile_string):
    profiles = profile_string.strip().split('\n\n')  # Split the string into profiles based on double newline
    user_profiles = []
    for profile in profiles:
        user_profile = profile.strip()
        user_profiles.append(user_profile)
    return user_profiles


def get_embedding(text, model="text-embedding-3-small"):
    global TOKENS
    TOKENS += len(text)
    return openai.Embedding.create(input=[text], model=model, dimensions=embedding_size).data[0].embedding


def generate_user_embeddings(user_profiles):
    user_embeddings = []
    for j, user in enumerate(user_profiles):
        emb = get_embedding(user[1])
        user_embeddings.append(emb)
    with open('./datasets/large/16_user_embeddings.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i, user_em in enumerate(user_embeddings):
            writer.writerow([user_profiles[i][0], user_em])
    return user_embeddings


def generate_arms_responses(user_profiles, num, name_prefix, llm):
    base_prompt = "Write a message for the following user to encourage them to donate to doctors without borders charity organization. "

    name = name_prefix + "arms-" + str(num) + ".csv"
    with open(name, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["user_id", "topic_id", "style_id", "arm_id", "response"])

    all_responses = dict()
    for i in range(len(user_profiles)):
        id = user_profiles[i][0]
        user = user_profiles[i][1]
        responses = []
        all_responses[id] = []
        print("arms for user:", i)
        for j, s in enumerate(styles):
            prompt = base_prompt + "Use the following style to generate the message: " + s
            prompt = prompt + "\n The user profile is the following:" + user
            response = get_chat_completion(prompt, llm)
            out_response = response.replace("\n", " ")
            row = [id, j, out_response]
            responses.append(row)
            all_responses[id].append(out_response)

        with open(name, 'a') as csv_file:
            writer = csv.writer(csv_file)
            for row in responses:
                writer.writerow(row)

    return all_responses


def generate_arms_context(llm):
    responses = []
    responses_embeddings = []
    base_prompt = "We have these styles to write an email to the users to encourage them to donate to the charity. For the following style, generate a user profile that is more likely to prefer that style. The styles is: "
    for i in range(len(styles)):
        prompt = base_prompt + " " + styles[i]
        response = get_chat_completion(prompt, llm)
        response = response.replace("\n", " ")
        responses.append(response)
        embedding = get_embedding(response)
        responses_embeddings.append(embedding)

    with open('arms-context.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for i in range(len(responses)):
            writer.writerow([responses[i]])

        for i in range(len(responses_embeddings)):
            writer.writerow([responses_embeddings[i]])


def generate_pair_rewards(user_profiles, arms, base, name_prefix, llm):
    arms_num = len(styles)
    for i, user_nid in enumerate(user_profiles):
        id = user_nid[0]
        user = user_nid[1]
        scores = [[-1 for _ in range(arms_num)] for _ in range(arms_num)]
        raws = []
        print(i, "------------------------")
        for j in range(arms_num):
            for jj in range(arms_num):
                if j == jj:
                    continue
                a1 = arms[id][j]
                a2 = arms[id][jj]
                print(id, j, jj, "##########")
                prompt = "pretend you are the following user: [USER]" + user + "[END] Now you are receiving these two messages: [Message 1]:" + a1 + " [END 1 ] [Message 2]:" + a2 + " [END 2 ] Which message is more alligned with your interests? Which one makes you donate to the charity? Let's think step by step. Print the number of preferred message at the end: [Answer]"
                response = get_chat_completion(prompt, llm)
                response = response.replace("\n", " ")
                x = max(len(response) - 20, 0)
                s = response[x:]
                num_response = -1
                if "1" in s:
                    num_response = 0
                elif "2" in s:
                    num_response = 1
                scores[j][jj] = num_response
                raws.append(response)

        with open(name_prefix + "rewards--" + str(base) + '.csv', 'a') as csv_file:
            writer = csv.writer(csv_file)
            for j in range(arms_num):
                for jj in range(arms_num):
                    if j == jj:
                        continue
                    ans = [id, j, jj, scores[j][jj]]
                    writer.writerow(ans)

        with open(name_prefix + "rewards--" + str(base) + '---raw.csv', 'a') as csv_file:
            writer = csv.writer(csv_file)
            for raw_id in range(len(raws)):
                ans = [id, raw_id, raws[raw_id]]
                writer.writerow(ans)


def generate_all(num, name_prefix, llm):
    base = num * 500
    for i in range(50):
        print("iter: ", i, "-----------")
        prompt = "Generate 10 detailed user profiles who we want to contact to donoate to doctors without borders charity organization. Add personal details to each, such as age, sex, location, occupation, hobbies, financial situation, reason to donate to charity, charities that they supported in the past, etc. Start each item with a * not a number. Follow the style of this: * John Smith, Male, 42, Seattle, Washington, USA. He is a Software Developer for a leading tech company. His hobbies include cycling, hiking, and reading about technology advancements. Although he is in a stable financial situation, he is very mindful about his expenditures. His primary reason to donate to charity is to help in making advancements in medical technology and healthcare. He has earlier donated to local health-focused NGOs."
        name = name_prefix + "users-" + str(num) + ".csv"
        response = get_chat_completion(prompt, llm)
        user_profiles = separate_profiles(response)
        write_user_profiles(user_profiles, name)
        print("user profiles done")
        arms = generate_arms_responses(user_profiles, base + (i * 10), name_prefix, llm)
        print("arms done")
        generate_pair_rewards(user_profiles, arms, base + (i * 10), name_prefix, llm)


prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
prs.add_argument("-p", dest="prefix", type=str, default="./datasets/large/", required=False, help="Name prefix.\n")
prs.add_argument("-n", dest="num", type=int, default=0, required=False, help="Run number.\n")
prs.add_argument("-m", dest="model", type=str, default="gpt3.5", required=False, help="LLM.\n")

args = prs.parse_args()
generate_all(args.num, args.prefix, args.model)
