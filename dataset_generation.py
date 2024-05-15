import pandas as pd
import numpy as np
from operator import itemgetter
import random


def incipit():
    return "Write an ASP program for the following problem."


def label_assignment(labels, predicate_name):
    f = []

    questions, answers = [], []

    n_max = 10

    n_labels = np.random.randint(2, n_max)
    labels_to_assign = np.random.choice(labels, size=n_labels, replace=False)
    question_assignment = f'''{incipit()} Assign exactly a label among a given set of labels to a set of elements. The set of elements is expressed by predicate {predicate_name}. The labels are {','.join([f"{x}" for x in labels_to_assign])}.'''

    answer_assignment = ""
    for label in labels_to_assign[:-1]:
        answer_assignment += f'''assign(X,"{label}")|'''
    answer_assignment += f'''assign(X,"{labels_to_assign[-1]}"):-{predicate_name}(X).'''

    questions.append(question_assignment)
    answers.append(answer_assignment)

    f.append(f"{predicate_name}(1..5).")

    return questions, answers, f


def prevent_value(labels, predicate_name):
    f = []
    fact = ''

    n_values = 20

    questions, answers = [], []

    value = np.random.randint(1, n_values)

    label = labels[np.random.randint(0, len(labels))]
    question_prevent_label = f'''{incipit()} Prevent the predicate {predicate_name} with value {value} from having label "{label}".'''
    answer_prevent_label = f''':-assign({value},"{label}").'''

    questions.append(question_prevent_label)
    answers.append(answer_prevent_label)

    fact += f'''{predicate_name}(1..{n_values}).'''

    for label in labels[:-1]:
        fact += f'''assign(X,"{label}")|'''

    fact += f'''assign(X,"{labels[-1]}"):-{predicate_name}(X).'''
    f.append(fact)

    return questions, answers, f


def generate_combinations(predicate_name_1, predicate_name_2):
    question = f'''{incipit()} Generate all the combinations of elements from two sets. The two sets are represented by predicates {predicate_name_1} and {predicate_name_2}.'''
    answer = f"combination(X,Y):-{predicate_name_1}(X),{predicate_name_2}(Y)."

    f = [f'''{predicate_name_1}(1..4).{predicate_name_2}(1..5).''']

    return [question], [answer], f


def select_value(predicate_name, label):
    question = f'''{incipit()} Select all values associated to the predicate {predicate_name} with label "{label}".'''
    answer = f'''select(X):-{predicate_name}(X,"{label}").'''

    f = [f'''{predicate_name}(1..5, "{label}").''']

    return [question], [answer], f


def execute_join(predicate_name_1, predicate_name_2, attributes):
    questions, answers = [], []
    f = []

    for attributes_1 in range(3, 6):
        for attributes_2 in range(2, 5):

            fact = ''

            n_attributes = attributes_1
            chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
            random_pos = np.random.randint(1, n_attributes)
            chosen_attributes[0] = f"ID"
            chosen_attributes[random_pos] = f"{predicate_name_2}ID"

            string_chosen_attributes = f'''{''.join([f'"{x}",' for x in chosen_attributes[:-1]])}'''
            string_chosen_attributes += f'"{chosen_attributes[-1]}"'
            fact += f'''{predicate_name_1}({string_chosen_attributes}).'''

            a = ''
            for attr in chosen_attributes[:-1]:
                a += f'"{attr}",'
            a += f'"{chosen_attributes[-1]}"'

            p = f"{predicate_name_1}("
            for i in range(len(chosen_attributes) - 1):
                if i == 0:
                    p += "X"
                elif i == random_pos:
                    p += "Y"
                else:
                    p += "_"

                p += ","

            if random_pos == len(chosen_attributes) - 1:
                p += "Y)"
            else:
                p += "_)"

            n_attributes = attributes_2
            chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
            chosen_attributes[0] = "ID"

            string_chosen_attributes_2 = f'''{''.join([f'"{x}",' for x in chosen_attributes[:-1]])}'''
            string_chosen_attributes_2 += f'"{chosen_attributes[-1]}"'
            fact += f'''{predicate_name_2}({string_chosen_attributes_2}).'''

            random_attribute_index = np.random.randint(1, n_attributes)
            random_attribute = chosen_attributes[random_attribute_index]

            b = ''
            for attr in chosen_attributes[:-1]:
                b += f'"{attr}",'
            b += f'"{chosen_attributes[-1]}"'

            question = f'''{incipit()} Consider predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates to each {predicate_name_1} the {random_attribute} of {predicate_name_2}.'''

            q = f"{predicate_name_2}("
            for i in range(len(chosen_attributes) - 1):
                if i == 0:
                    q += "Y"
                elif i == random_attribute_index:
                    q += "Z"
                else:
                    q += "_"

                q += ","

            if random_attribute_index == len(chosen_attributes) - 1:
                q += "Z)"
            else:
                q += "_)"

            answer = f'''{predicate_name_1}_{predicate_name_2}(X,Z):-{p},{q}.'''

            questions.append(question)
            answers.append(answer)
            f.append(fact)

    return questions, answers, f


def transitive_closure(closure_name, predicate_name):
    question = f'''{incipit()} Define predicate "{closure_name}" as the transitive closure of predicate "{predicate_name}".'''

    answer = f'''{closure_name}(X,Y):-{predicate_name}(X,Y).\n{closure_name}(X,Y):-{predicate_name}(X,Z),{closure_name}(Z,Y).'''

    f = [f'''{predicate_name}(1..3, 1..4).''']

    return [question], [answer], f


def preferences(predicate_name, labels):
    questions, answers, f = [], [], []
    n_values = 20

    for cost_value in range(1, 3):
        for cost_level in range(1, 3):
            value = np.random.randint(1, n_values)

            label = labels[np.random.randint(0, len(labels))]
            question_preference = f'''{incipit()} I would prefer that predicate {predicate_name} with value {value} is not associated with "{label}". If this occurs, it costs {cost_value} at level {cost_level}.'''
            answer_preference = f''':~assign({value},"{label}").[{cost_value}@{cost_level}]'''

            questions.append(question_preference)
            answers.append(answer_preference)

    fact = f'''{predicate_name}(1..{n_values}).'''

    for label in labels[:-1]:
        fact += f'''assign(X,"{label}")|'''

    fact += f'''assign(X,"{labels[-1]}"):-{predicate_name}(X).'''

    f.append(fact)

    return questions, answers, f


def select_by_negative_condition(predicate_name, not_predicate_name, labels):
    label = labels[np.random.randint(0, len(labels))]

    question = f'''{incipit()} Select all values associated with predicate {predicate_name} but not associated with predicate {not_predicate_name} and label "{label}".'''
    answer = f'''select(X):-{predicate_name}(X),not {not_predicate_name}(X,"{label}").'''

    chosen_labels = list(set(list(np.random.choice(labels, size=4, replace=False))).union({label}))
    combinations = list(zip(range(1, 4), chosen_labels))

    fact = f'''{predicate_name}(1..3).'''

    for i, l in combinations:
        fact += f'''{not_predicate_name}({i},"{l}").'''

    return [question], [answer], [fact]


def select_by_numeric_condition(predicate_name):
    # condition \in [!=, <, >, <=, >=]

    n_values = 100

    condition_dict = {"different": "!=", "greater": ">", "lower": "<", "greater or equal": ">=", "lower or equal": "<="}

    questions, answers = [], []

    for condition, condition_symbol in condition_dict.items():
        condition_value = np.random.randint(1, n_values)

        question = f'''{incipit()} Select all values associated with predicate {predicate_name} with a value {condition} than {condition_value}.'''
        answer = f'''select(X):-{predicate_name}(X,C),C{condition_symbol}{condition_value}.'''

        questions.append(question)
        answers.append(answer)

    f = [f'''{predicate_name}(1..3, 1..{n_values}).''']

    return questions, answers, f


def generate_subproblems(size, train_size, validation, print_proportions=False):
    colors = ["red", "green", "blue", "yellow", "brown", "orange", "purple", "gray", "cyan"]
    cities = ["rome", "paris", "venice", "new york", "london", "amsterdam", "dubai", "tokyo", "shangai", "florence"]
    labels = ["color", "person", "tree", "car", "moto", "bike", "table", "food", "element", "street", "object"]
    attributes = ["price", "name", "city", "age", "author", "creator", "shape", "height", "description"]

    predicates = colors + cities + labels + attributes
    closures = ["path", "flights", "ancestors", "destinations", "arrivals"]

    questions = []
    answers = []

    for i in tqdm(range(size), total=size):
        if not validation:
            np.random.seed(i)
        else:
            np.random.seed(train_size + i)

        for _ in range(10):
            question_assignments, answer_assignments, f = label_assignment(predicates, np.random.choice(predicates))
            questions.extend(question_assignments)
            answers.extend(answer_assignments)

        n_questions_assignment = len(questions)

        for _ in range(5):
            question_prevents, answer_prevents, f = prevent_value(predicates, np.random.choice(predicates))
            questions.extend(question_prevents)
            answers.extend(answer_prevents)

        n_questions_prevent = len(questions) - n_questions_assignment

        p_1, p_2 = np.random.choice(predicates, 2, replace=False)
        questions_combinations, answers_combinations, f = generate_combinations(p_1, p_2)

        questions.extend(questions_combinations)
        answers.extend(answers_combinations)

        questions_select, answers_select, f = select_value(np.random.choice(predicates), np.random.choice(predicates))

        questions.extend(questions_select)
        answers.extend(answers_select)

        p_1, p_2 = np.random.choice(predicates, 2, replace=False)
        questions_join, answers_join, f = execute_join(p_1, p_2, attributes)

        questions.extend(questions_join)
        answers.extend(answers_join)

        questions_closure, answers_closure, f = transitive_closure(np.random.choice(closures),
                                                                   np.random.choice(predicates))

        questions.extend(questions_closure)
        answers.extend(answers_closure)

        questions_preferences, answers_preferences, f = preferences(np.random.choice(predicates), predicates)

        questions.extend(questions_preferences)
        answers.extend(answers_preferences)
        facts.extend(f)

        questions_negative, answers_negative, f = select_by_negative_condition(np.random.choice(predicates),
                                                                               np.random.choice(predicates), predicates)

        questions.extend(questions_negative)
        answers.extend(answers_negative)

        questions_numeric_condition, answers_numeric_condition, f = select_by_numeric_condition(
            np.random.choice(predicates))

        questions.extend(questions_numeric_condition)
        answers.extend(answers_numeric_condition)

        if print_proportions:
            print("N questions assignment:", n_questions_assignment * size,
                  n_questions_assignment * size / len_df * 100)
            print("N questions prevent:", n_questions_prevent * size, n_questions_prevent * size / len_df * 100)
            print("N questions combinations:", len(questions_combinations) * size,
                  len(questions_combinations) * size / len_df * 100)
            print("N questions select:", len(questions_select) * size, len(questions_select) * size / len_df * 100)
            print("N questions join:", len(questions_join) * size, len(questions_join) * size / len_df * 100)
            print("N questions closure:", len(questions_closure) * size, len(questions_closure) * size / len_df * 100)
            print("N questions preferences:", len(questions_preferences) * size,
                  len(questions_preferences) * size / len_df * 100)
            print("N questions negative:", len(questions_negative) * size,
                  len(questions_negative) * size / len_df * 100)
            print("N questions numeric condition:", len(questions_numeric_condition) * size,
                  len(questions_numeric_condition) * size / len_df * 100)
            break

    random.seed(42)
    temp = list(zip(questions, answers))

    random.shuffle(temp)
    res1, res2 = zip(*temp)

    questions, answers = list(res1), list(res2)

    return questions, answers


tot_size = 100000
size = int(0.8 * tot_size)
val_size = int(0.2 * tot_size)

questions, answers = generate_subproblems(size, size, validation=False)
val_questions, val_answers = generate_subproblems(val_size, size, validation=True)

d = {"question": questions, "answer": answers}
val_d = {"question": val_questions, "answer": val_answers}

train_df = pd.DataFrame(d)
val_df = pd.DataFrame(val_d)

# CHANGE PATH IF NEEDED
train_df_fn = "train_df.csv"
val_df_fn = "val_df.csv"

train_df.to_csv(train_df_fn, index=False)
val_df.to_csv(val_df_fn, index=False)