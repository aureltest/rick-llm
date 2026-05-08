from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import os
import time

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL")          # None → OpenAI, sinon ex: http://localhost:11434/v1
LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-4o-mini")

RICK_SYSTEM_PROMPT = """Tu incarnes Rick Sanchez, un scientifique de génie interdimensionnel.
Fais preuve d'une honnêteté sans concession, d'un esprit vif et saupoudre tes propos de jargon scientifique.
N'hésite pas à recourir à l'humour noir ou à aborder des vérités existentielles, mais propose toujours une solution (même si elle sort des sentiers battus)."""

CLEANING_PROMPT = """Tu vas recevoir des répliques de dialogues en anglais issues de transcriptions de séries.
Ta mission est de :
1. Supprimer les didascalies, actions entre parenthèses ou crochets, et toute description de contexte
2. Supprimer les symboles incorrects (deux-points en début de phrase, etc.)
3. Traduire le résultat en français naturel et courant

Réponds UNIQUEMENT avec la réplique finale en français, sans explication.

Exemples :

Entrée : stumbles in drunkenly, and turns on the lights. Morty! You gotta come on. Jus'... you gotta come with me.
Résultat : Morty ! Tu dois venir. Il faut juste... que tu viennes avec moi.

Entrée : rubs his eyes. What, Rick? What's going on?
Résultat : Quoi, Rick ? Qu'est-ce qui se passe ?

Entrée : Oh, I don't know, maybe because you're not a scientist?
Résultat : Oh, je ne sais pas, peut-être parce que tu n'es pas un scientifique ?"""


def load_rick_and_morty_dataset():
    """Loads the Rick and Morty transcript dataset.

    This function loads the Rick and Morty transcript dataset from the Hugging Face
    datasets hub, specifically from the "Prarabdha/Rick_and_Morty_Transcript" dataset.

    Returns:
        datasets.Dataset: A dataset containing Rick and Morty episode transcripts.
            The dataset includes columns for dialogue, speaker, and episode information.
    """
    dataset = load_dataset("Prarabdha/Rick_and_Morty_Transcript", split="train")
    return dataset

def create_conversation_pairs(dataset):
    """Creates conversation pairs from the Rick and Morty transcript dataset.

    This function processes the dataset to create conversation pairs where a non-Rick character
    speaks followed by Rick's response. Each conversation includes a system prompt defining
    Rick's character.

    Args:
        dataset (datasets.Dataset): The Rick and Morty transcript dataset containing dialogue
            and speaker information.

    Returns:
        datasets.Dataset: A new dataset containing conversation pairs in the format:
            {
                "conversations_raw": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": non_rick_dialogue},
                    {"from": "gpt", "value": rick_dialogue}
                ]
            }
    """
    new_rows = []
    for i in tqdm(range(len(dataset) - 1)):
        current_row = dataset[i]
        next_row = dataset[i + 1]

        if current_row["speaker"] != "Rick" and next_row["speaker"] == "Rick":
            if current_row["episode no."] == next_row["episode no."]:
                new_rows.append(
                    {
                        "conversations_raw": [
                            {"from": "system", "value": RICK_SYSTEM_PROMPT.strip()},
                            {"from": "human", "value": current_row["dialouge"].strip()},
                            {"from": "gpt", "value": next_row["dialouge"].strip()},
                        ]
                    }
                )

    return Dataset.from_list(new_rows)

def clean_dialogue(client, text, system_prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text.strip()},
                ],
                temperature=0.2,
                max_tokens=256,
            )
            return response.choices[0].message.content
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

def clean_conversations(dataset):
    """Clean all conversations in the dataset by removing action descriptions and context.

    This function processes each conversation in the dataset by removing action descriptions,
    stage directions, and other contextual information from the dialogue, leaving only the
    spoken lines.

    Args:
        dataset (datasets.Dataset): The input dataset containing conversations in the format:
            {
                "conversations_raw": [
                    {"from": "system", "value": str},
                    {"from": "human", "value": str},
                    {"from": "gpt", "value": str}
                ]
            }

    Returns:
        datasets.Dataset: A new dataset with cleaned conversations in the format:
            {
                "conversations": [
                    {"from": "system", "value": str},
                    {"from": "human", "value": str},
                    {"from": "gpt", "value": str}
                ]
            }
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=LLM_BASE_URL,
    )
    new_rows = []
    seen = set()

    for row in tqdm(dataset):
        human_clean = clean_dialogue(
            client, row["conversations_raw"][1]["value"], CLEANING_PROMPT
        )
        rick_clean = clean_dialogue(
            client, row["conversations_raw"][2]["value"], CLEANING_PROMPT
        )

        if not human_clean or not rick_clean:
            continue
        human_clean = human_clean.strip()
        rick_clean = rick_clean.strip()
        if len(human_clean) < 2 or len(rick_clean) < 2:
            continue

        key = (human_clean, rick_clean)
        if key in seen:
            continue
        seen.add(key)

        new_rows.append(
            {
                "conversations": [
                    {"from": "system", "value": row["conversations_raw"][0]["value"]},
                    {"from": "human", "value": human_clean},
                    {"from": "gpt", "value": rick_clean},
                ]
            }
        )

    return Dataset.from_list(new_rows)


def main():
    backend = LLM_BASE_URL or "https://api.openai.com/v1 (OpenAI)"
    print(f"LLM backend : {backend}")
    print(f"LLM model   : {LLM_MODEL}")
    print("Loading dataset...")
    dataset = load_rick_and_morty_dataset()
    print("Number of rows: ", len(dataset))

    print("Creating conversation pairs...")
    sharegpt_dataset = create_conversation_pairs(dataset)

    print("Cleaning conversations...")
    cleaned_dataset = clean_conversations(sharegpt_dataset)

    print("Saving locally to ./dataset_cache ...")
    cleaned_dataset.save_to_disk("./dataset_cache")

    print("Pushing to hub...")
    cleaned_dataset.push_to_hub(
        os.getenv("HF_DATASET_NAME", "your-username/rick-et-morty-transcripts-fr-sharegpt"),
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )
    print("Done!")


if __name__ == "__main__":
    main()



