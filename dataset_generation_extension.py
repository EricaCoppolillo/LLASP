from huggingface_hub import login
import os
from tqdm import tqdm
import sys
import torch
import pandas as pd

import numpy as np
from clingo.symbol import parse_term
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
hugging_token = "[HUGGINGFACE TOKEN]"
login(hugging_token)

torch.cuda.is_available()

torch.cuda.device_count()

torch.manual_seed(56)


class Context:
    # get features/words from a string of space separated words
    def gen_feature(self, x):
        ret = []
        for term in str(x.string).split(' '):
            ret.append(parse_term(term))
        return ret


def incipit():
    return "Write an ASP program for the following problem."


def rephrased_label_assignments(labels, predicate_name, test):
    labels_set = ''.join([f'"{x}",' for x in labels[:-1]]) + f'"{labels[-1]}"'

    if not test:
        s1 = f'''Develop an ASP program that assigns exactly one label from the specified set {labels_set} to a collection of elements defined by the predicate "{predicate_name}".'''

        s2 = f'''Create an ASP application that maps one label belonging to {labels_set} to a set of elements based on the predicate "{predicate_name}".'''

        s3 = f'''Write an ASP script that associates exactly one label from the set {labels_set} with a group of elements determined by the predicate "{predicate_name}".'''

        s4 = f'''Design an ASP solution that links a single label from {labels_set} to elements specified by the predicate "{predicate_name}".'''

        s5 = f'''Craft an ASP program that assigns one label from {labels_set} to elements defined by the predicate "{predicate_name}".'''

        s6 = f'''Implement an ASP code snippet that tags elements with one label from the set {labels_set} according to the predicate "{predicate_name}".'''

        s7 = f'''Build an ASP application that links exactly one label from {labels_set} to a set of elements identified by the predicate "{predicate_name}".'''

        s8 = f'''Write an ASP script that connects a single label from {labels_set} to each element defined by the predicate "{predicate_name}".'''

        s9 = f'''Develop an ASP solution to map one label from {labels_set} to elements as per the predicate "{predicate_name}".'''

        s10 = f'''Create an ASP program that assigns just one label from {labels_set} to a collection of elements determined by the predicate "{predicate_name}".'''

        s11 = f'''Formulate an ASP solution that links a single label from {labels_set} with elements identified by the predicate "{predicate_name}".'''

        s12 = f'''Compose an ASP script to link only one label from {labels_set} to a group of elements according to the predicate "{predicate_name}".'''

        s13 = f'''Draft an ASP program that maps a single label from {labels_set} to elements defined by the predicate "{predicate_name}".'''

        s14 = f'''Generate an ASP code that attaches one label from {labels_set} to elements specified by the predicate "{predicate_name}".'''

        s15 = f'''Develop an ASP application that links a single label from {labels_set} with elements as indicated by the predicate "{predicate_name}".'''

        s16 = f'''Create an ASP script that connects one label from {labels_set} to elements based on the predicate "{predicate_name}".'''

        s17 = f'''Build an ASP program that maps a single label from {labels_set} to elements guided by the predicate "{predicate_name}".'''

        s18 = f'''Write an ASP solution that tags elements with a single label from {labels_set} according to the predicate "{predicate_name}".'''

        s19 = f'''Design an ASP script to link exactly one label from {labels_set} to elements under the predicate "{predicate_name}".'''

        s20 = f'''Compose an ASP application that assigns exactly one label from {labels_set} to elements determined by the predicate "{predicate_name}".'''

        s = []

        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:

        s1 = f'''Develop an ASP script that ensures each element, as specified by the predicate "{predicate_name}", receives exactly one label from the set {labels_set}.'''

        s2 = f'''Create an ASP solution to assign one specific label from {labels_set} to a group of elements as defined by the predicate "{predicate_name}".'''

        s3 = f'''Write an ASP application that maps a single label from {labels_set} to every element identified by the predicate "{predicate_name}".'''

        s4 = f'''Design an ASP script to connect each element, as determined by the predicate "{predicate_name}", with one label from {labels_set}.'''

        s5 = f'''Craft an ASP solution that associates precisely one label from {labels_set} with elements specified by the predicate "{predicate_name}".'''

        s6 = f'''Implement an ASP application to tag elements, defined by the predicate "{predicate_name}", with one label from the set {labels_set}.'''

        s7 = f'''Build an ASP program that links each element identified by the predicate "{predicate_name}" to a single label from {labels_set}.'''

        s8 = f'''Write an ASP code snippet to connect a single label from {labels_set} to elements specified by the predicate "{predicate_name}".'''

        s9 = f'''Develop an ASP solution to map one specific label from {labels_set} to each element defined by the predicate "{predicate_name}".'''

        s10 = f'''Create an ASP script that assigns a single label from {labels_set} to a group of elements as indicated by the predicate "{predicate_name}".'''

        s11 = f'''Formulate an ASP program that links each element, as identified by the predicate "{predicate_name}", with one label from {labels_set}.'''

        s12 = f'''Compose an ASP application that assigns one label from {labels_set} to every element defined by the predicate "{predicate_name}".'''

        s13 = f'''Draft an ASP code that connects a single label from the set {labels_set} to elements specified by the predicate "{predicate_name}".'''

        s14 = f'''Generate an ASP solution that links one label from {labels_set} with each element identified by the predicate "{predicate_name}".'''

        s15 = f'''Develop an ASP application to assign one label from {labels_set} to elements defined by the predicate "{predicate_name}".'''

        s16 = f'''Create an ASP script that maps a single label from {labels_set} to a collection of elements specified by the predicate "{predicate_name}".'''

        s17 = f'''Build an ASP code snippet to link one label from {labels_set} to elements identified by the predicate "{predicate_name}".'''

        s18 = f'''Write an ASP solution to connect each element defined by the predicate "{predicate_name}" with a single label from {labels_set}.'''

        s19 = f'''Design an ASP application to assign one label from {labels_set} to every element specified by the predicate "{predicate_name}".'''

        s20 = f'''Compose an ASP program that maps a single label from the set {labels_set} to elements determined by the predicate "{predicate_name}".'''

        s21 = f'''{incipit()} Assign exactly a label among a given set of labels to a set of elements. The set of elements is expressed by predicate {predicate_name}. The labels are {labels_set}.'''

        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s


def label_assignment(labels, predicate_name, prompt_invariance, test):
    f = []

    questions, answers = [], []
    rewritten_questions = []

    n_max = 10

    n_labels = np.random.randint(2, n_max)
    labels_to_assign = np.random.choice(labels, size=n_labels,
                                        replace=False)  # dalla raccolta di etichette ne sceglie "size" e senza rimpiazzo   Crea quindi un array di dimensione size scegliendo casualmente gli elementi da "labels"
    question = f'''{incipit()} Assign exactly a label among a given set of labels to a set of elements. The set of elements is expressed by predicate {predicate_name}. The labels are {','.join([f"{x}" for x in labels_to_assign])}.'''

    if (prompt_invariance):
        rewritten_questions = rephrased_label_assignments(labels_to_assign, predicate_name, test)

    answer = ""
    for label in labels_to_assign[:-1]:  # si ferma al penultimo elemento perché l'ultimo verrà messo manualmente sotto
        answer += f'''assign(X,"{label}")|'''
    answer += f'''assign(X,"{labels_to_assign[-1]}"):-{predicate_name}(X).'''

    rewritten_answers = [answer] * len(rewritten_questions)

    f = f"{predicate_name}(1..5)."

    if (len(rewritten_questions) > 0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    return questions, answers, f


def rephrased_prevent_value(predicate_name, value, label, test):
    if not test:
        s1 = f'''Create an ASP program that ensures the predicate "{predicate_name}" with a value of {value} is not assigned to the label "{label}".'''

        s2 = f'''Write an ASP solution that prohibits the assignment of the predicate "{predicate_name}" with a value of {value} to the label "{label}".'''

        s3 = f'''Develop an ASP program that disallows associating the predicate "{predicate_name}" having value {value} with the label "{label}".'''

        s4 = f'''Craft an ASP script that prevents the predicate "{predicate_name}" with value {value} from being linked to the label "{label}".'''

        s5 = f'''Implement an ASP application that avoids assigning the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s6 = f'''Design an ASP solution that excludes the predicate "{predicate_name}" with value {value} from being mapped to the label "{label}".'''

        s7 = f'''Build an ASP program that ensures the predicate "{predicate_name}" having value {value} cannot be assigned to the label "{label}".'''

        s8 = f'''Create an ASP code snippet that disallows associating the predicate "{predicate_name}" having value {value} with the label "{label}".'''

        s9 = f'''Write an ASP program that prevents the predicate "{predicate_name}" with value {value} from being linked to the label "{label}".'''

        s10 = f'''Develop an ASP solution that avoids assigning the predicate "{predicate_name}" having value {value} to the label "{label}".'''

        s11 = f'''Formulate an ASP program that ensures the predicate "{predicate_name}" with a value of {value} is not linked to the label "{label}".'''

        s12 = f'''Compose an ASP script that prohibits the predicate "{predicate_name}" with a value of {value} from being assigned to the label "{label}".'''

        s13 = f'''Draft an ASP solution that disallows linking the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s14 = f'''Generate an ASP application that prevents the predicate "{predicate_name}" having value {value} from being mapped to the label "{label}".'''

        s15 = f'''Develop an ASP script that avoids assigning the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s16 = f'''Create an ASP solution that excludes the predicate "{predicate_name}" with a value of {value} from the label "{label}".'''

        s17 = f'''Build an ASP program that disallows associating the predicate "{predicate_name}" having value {value} with the label "{label}".'''

        s18 = f'''Write an ASP application that ensures the predicate "{predicate_name}" with value {value} is not assigned to the label "{label}".'''

        s19 = f'''Design an ASP script that prohibits the assignment of the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s20 = f'''Create an ASP program that prevents the predicate "{predicate_name}" with a value of {value} from being linked to the label "{label}".'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:

        s1 = f'''Develop an ASP application that avoids the predicate "{predicate_name}" with a value of {value} being linked to the label "{label}".'''

        s2 = f'''Compose an ASP solution to ensure the predicate "{predicate_name}" with value {value} is not associated with the label "{label}".'''

        s3 = f'''Create an ASP script that excludes the predicate "{predicate_name}" with value {value} from being mapped to the label "{label}".'''

        s4 = f'''Generate an ASP application to prevent linking the predicate "{predicate_name}" with a value of {value} to the label "{label}".'''

        s5 = f'''Draft an ASP program to disallow assigning the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s6 = f'''Formulate an ASP code that ensures the predicate "{predicate_name}" having value {value} is not connected to the label "{label}".'''

        s7 = f'''Produce an ASP program that prevents associating the predicate "{predicate_name}" with value {value} with the label "{label}".'''

        s8 = f'''Build an ASP solution that disallows the predicate "{predicate_name}" having value {value} from being assigned to the label "{label}".'''

        s9 = f'''Craft an ASP application to avoid mapping the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s10 = f'''Create an ASP code snippet to ensure the predicate "{predicate_name}" with a value of {value} is not linked to the label "{label}".'''

        s11 = f'''Write an ASP script that prevents the predicate "{predicate_name}" with value {value} from being assigned to the label "{label}".'''

        s12 = f'''Develop an ASP application to disallow connecting the predicate "{predicate_name}" having value {value} with the label "{label}".'''

        s13 = f'''Compose an ASP solution that avoids the predicate "{predicate_name}" with value {value} being mapped to the label "{label}".'''

        s14 = f'''Generate an ASP code to exclude linking the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s15 = f'''Formulate an ASP script to ensure the predicate "{predicate_name}" having value {value} is not associated with the label "{label}".'''

        s16 = f'''Design an ASP application that prohibits assigning the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s17 = f'''Produce an ASP solution that disallows the predicate "{predicate_name}" with value {value} from being mapped to the label "{label}".'''

        s18 = f'''Create an ASP script to avoid associating the predicate "{predicate_name}" having value {value} with the "{label}" label.'''

        s19 = f'''Draft an ASP program to prevent the predicate "{predicate_name}" with value {value} from being linked to the label "{label}".'''

        s20 = f'''Write an ASP application that excludes the predicate "{predicate_name}" with value {value} from being assigned to the label "{label}".'''

        s21 = f'''{incipit()} Prevent the predicate "{predicate_name}" with value "{value}" from having label "{label}".'''

        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s


def prevent_value(labels, predicate_name, prompt_invariance, test):
    f = []
    fact = ''

    n_values = 20

    questions, answers = [], []
    rewritten_questions = []

    value = np.random.randint(1, n_values)

    label = labels[np.random.randint(0, len(labels))]
    question = f'''{incipit()} Prevent the predicate "{predicate_name}" with value "{value}" from having label "{label}".'''

    if (prompt_invariance):
        rewritten_questions = rephrased_prevent_value(predicate_name, value, label, test)

    answer = f''':-assign({value},{label}).'''
    rewritten_answers = [answer] * len(rewritten_questions)

    if (len(rewritten_questions) > 0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    fact += f'''{predicate_name}(1..{n_values}).'''
    for label in labels[:-1]:
        fact += f'''assign(X,"{label}")|'''

    fact += f'''assign(X,"{labels[-1]}"):-{predicate_name}(X).'''

    return questions, answers, fact


def rephrased_generate_combinations(predicate_name_1, predicate_name_2, test):
    if not test:
        s1 = f'''Develop an ASP program that computes all possible combinations of elements from two sets represented by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s2 = f'''Write an ASP solution that generates the cross-product of elements between the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s3 = f'''Create an ASP program that produces all valid pairings of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s4 = f'''Design an ASP script that calculates the Cartesian product of elements between the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s5 = f'''Implement an ASP application that finds all combinations of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s6 = f'''Craft an ASP solution that enumerates every possible pairing of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s7 = f'''Build an ASP program that lists all valid combinations of elements between the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s8 = f'''Create an ASP code snippet that computes the cross-product of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s9 = f'''Write an ASP program that generates all valid pairings of elements expressed by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s10 = f'''Develop an ASP solution that calculates the Cartesian product of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s11 = f'''Compose an ASP program that determines all possible combinations of elements from two sets defined by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s12 = f'''Generate an ASP solution that produces the cross-product of elements between the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s13 = f'''Create an ASP script that forms all valid pairings of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s14 = f'''Design an ASP program that computes the Cartesian product of elements from the sets represented by "{predicate_name_1}" and "{predicate_name_2}".'''

        s15 = f'''Implement an ASP solution that finds combinations of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s16 = f'''Craft an ASP script that enumerates all possible pairings of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s17 = f'''Develop an ASP code that lists valid combinations of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s18 = f'''Write an ASP snippet that computes the cross-product of elements in the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s19 = f'''Generate an ASP application that creates all valid pairings of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s20 = f'''Compose an ASP solution to calculate the Cartesian product of elements in the sets represented by "{predicate_name_1}" and "{predicate_name_2}".'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:

        s1 = f'''Design an ASP solution to compute all possible pairings of elements from two sets defined by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s2 = f'''Craft an ASP program to generate the cross-product of elements between the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s3 = f'''Develop an ASP code snippet to produce all valid combinations of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s4 = f'''Compose an ASP script to calculate the Cartesian product of elements represented by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s5 = f'''Write an ASP application that finds all pairings of elements from the sets defined by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s6 = f'''Formulate an ASP program that enumerates every possible combination of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s7 = f'''Create an ASP solution to list all valid pairings of elements between the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s8 = f'''Generate an ASP code to compute the cross-product of elements in the sets defined by "{predicate_name_1}" and "{predicate_name_2}".'''

        s9 = f'''Develop an ASP script to produce all valid pairings of elements as defined by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s10 = f'''Craft an ASP application that calculates the Cartesian product of elements between the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s11 = f'''Write an ASP program that determines all possible combinations of elements from sets represented by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s12 = f'''Compose an ASP script that generates the cross-product of elements between the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s13 = f'''Formulate an ASP code snippet to form all valid pairings of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s14 = f'''Create an ASP program to calculate the Cartesian product of elements from sets represented by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s15 = f'''Develop an ASP solution that finds all pairings of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s16 = f'''Generate an ASP script to enumerate all possible pairings of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s17 = f'''Craft an ASP application to list valid combinations of elements between the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s18 = f'''Write an ASP program that computes the cross-product of elements in the sets defined by "{predicate_name_1}" and "{predicate_name_2}".'''

        s19 = f'''Produce an ASP script to generate all valid pairings of elements as represented by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s20 = f'''Create an ASP solution to calculate the Cartesian product of elements from sets defined by "{predicate_name_1}" and "{predicate_name_2}".'''

        s21 = f'''{incipit()} Generate all the combinations of elements from two sets. The two sets are represented by predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s


def generate_combinations(predicate_name_1, predicate_name_2, prompt_invariance, test):
    questions, answers = [], []
    rewritten_questions = []

    question = f'''{incipit()} Generate all the combinations of elements from two sets. The two sets are represented by predicates "{predicate_name_1}" and "{predicate_name_2}".'''

    if (prompt_invariance):
        rewritten_questions = rephrased_generate_combinations(predicate_name_1, predicate_name_2, test)

    answer = f"combination(X,Y):-{predicate_name_1}(X),{predicate_name_2}(Y)."
    rewritten_answers = [answer] * len(rewritten_questions)

    if (len(rewritten_questions) > 0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    f = f'''{predicate_name_1}(1..4).{predicate_name_2}(1..5).'''

    return questions, answers, f


def rephrased_select_value(predicate_name, label, test):
    if not test:
        s1 = f'''Create an ASP program that retrieves all values associated with the predicate "{predicate_name}" labeled as "{label}".'''

        s2 = f'''Write an ASP program to extract values linked to the predicate "{predicate_name}" with the label "{label}".'''

        s3 = f'''Develop an ASP solution that identifies all values related to the label "{label}" within the predicate "{predicate_name}".'''

        s4 = f'''Craft an ASP program that collects data associated with the label "{label}" for the predicate "{predicate_name}".'''

        s5 = f'''Construct an ASP script to fetch values corresponding to the label "{label}" within the predicate "{predicate_name}".'''

        s6 = f'''Generate an ASP code snippet that retrieves all relevant values for the label "{label}" in the context of the predicate "{predicate_name}".'''

        s7 = f'''Produce an ASP implementation that selects values tied to the label "{label}" under the predicate "{predicate_name}".'''

        s8 = f'''Write an ASP script to obtain all values labeled as "{label}" within the predicate "{predicate_name}".'''

        s9 = f'''Design an ASP program to capture values associated with the label "{label}" in the context of the predicate "{predicate_name}".'''

        s10 = f'''Compose an ASP solution that identifies and retrieves values labeled "{label}" under the predicate "{predicate_name}".'''

        s11 = f'''Formulate an ASP script to gather values related to the label "{label}" within the predicate "{predicate_name}".'''

        s12 = f'''Create an ASP program that extracts values linked to the "{predicate_name}" predicate and labeled as "{label}".'''

        s13 = f'''Develop an ASP script to collect data associated with the "{predicate_name}" predicate and the label "{label}".'''

        s14 = f'''Design an ASP application that retrieves values associated with the label "{label}" within the predicate "{predicate_name}".'''

        s15 = f'''Write an ASP code snippet to fetch values linked to the label "{label}" in the context of the predicate "{predicate_name}".'''

        s16 = f'''Generate an ASP program that identifies all values associated with the label "{label}" within the pred icate "{predicate_name}".'''

        s17 = f'''Craft an ASP application to gather all values tied to the label "{label}" under the predicate "{predicate_name}".'''

        s18 = f'''Produce an ASP script that extracts values related to the label "{label}" in the context of the predicate "{predicate_name}".'''

        s19 = f'''Develop an ASP program that retrieves data associated with the label "{label}" within the predicate "{predicate_name}".'''

        s20 = f'''Create an ASP solution to capture values labeled as "{label}" within the predicate "{predicate_name}".'''

        s = []

        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:

        s1 = f'''Formulate an ASP application to fetch all values tied to the predicate "{predicate_name}" and labeled as "{label}".'''

        s2 = f'''Draft an ASP code to retrieve values associated with the predicate "{predicate_name}" and the label "{label}".'''

        s3 = f'''Generate an ASP script that identifies all values within the predicate "{predicate_name}" that are linked to the label "{label}".'''

        s4 = f'''Compose an ASP solution to gather data from the predicate "{predicate_name}" associated with the label "{label}".'''

        s5 = f'''Develop an ASP program to select values tied to the label "{label}" within the predicate "{predicate_name}".'''

        s6 = f'''Craft an ASP code snippet to capture all relevant values for the label "{label}" within the predicate "{predicate_name}".'''

        s7 = f'''Write an ASP script to collect values associated with the label "{label}" from the predicate "{predicate_name}".'''

        s8 = f'''Create an ASP solution that retrieves all values labeled "{label}" within the predicate "{predicate_name}".'''

        s9 = f'''Design an ASP application to fetch values tied to the label "{label}" within the context of the predicate "{predicate_name}".'''

        s10 = f'''Produce an ASP program to gather and retrieve values linked to the label "{label}" in the predicate "{predicate_name}".'''

        s11 = f'''Formulate an ASP script that extracts values related to the label "{label}" within the context of the predicate "{predicate_name}".'''

        s12 = f'''Write an ASP application to collect values linked to the predicate "{predicate_name}" and labeled as "{label}".'''

        s13 = f'''Develop an ASP solution that gathers data associated with the label "{label}" within the predicate "{predicate_name}".'''

        s14 = f'''Generate an ASP code snippet to capture values related to the label "{label}" in the predicate "{predicate_name}".'''

        s15 = f'''Compose an ASP program to identify values labeled as "{label}" within the predicate "{predicate_name}".'''

        s16 = f'''Craft an ASP application to fetch all values linked to the label "{label}" in the context of the predicate "{predicate_name}".'''

        s17 = f'''Design an ASP program to gather values tied to the label "{label}" within the context of the predicate "{predicate_name}".'''

        s18 = f'''Create an ASP code to retrieve values associated with the label "{label}" within the predicate "{predicate_name}".'''

        s19 = f'''Develop an ASP script to capture all values linked to the label "{label}" within the predicate "{predicate_name}".'''

        s20 = f'''Write an ASP solution to collect values tied to the predicate "{predicate_name}" and labeled as "{label}".'''

        s21 = f'''{incipit()} Select all values associated to the predicate "{predicate_name}" with label "{label}".'''

        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s


def select_value(predicate_name, label, prompt_invariance, test):
    questions, answers = [], []
    rewritten_questions = []

    question = f'''{incipit()} Select all values associated to the predicate "{predicate_name}" with label "{label}".'''

    if (prompt_invariance):
        rewritten_questions = rephrased_select_value(predicate_name, label, test)

    answer = f'''select(X):-{predicate_name}(X,"{label}").'''
    rewritten_answers = [answer] * len(rewritten_questions)

    if (len(rewritten_questions) > 0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    f = f'''{predicate_name}(1..5, "{label}").'''

    return questions, answers, f


def rephrased_execute_join(predicate_name_1, predicate_name_2, a, b, random_attribute, test):
    if not test:
        s1 = f'''Write an ASP program for the following problem. Consider predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates to each {predicate_name_1} the {random_attribute} of {predicate_name_2}.'''

        s2 = f'''Develop an ASP program for the problem described. Use the predicate "{predicate_name_1}" with fields {a} and the predicate "{predicate_name_2}" with fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s3 = f'''Create an ASP program for the following task. The predicate "{predicate_name_1}" has fields {a}, and the predicate "{predicate_name_2}" has fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s4 = f'''Construct an ASP program to solve the given problem. Consider the predicate "{predicate_name_1}" with fields {a} and the predicate "{predicate_name_2}" with fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s5 = f'''Draft an ASP program for the problem at hand. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that matches each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s6 = f'''Formulate an ASP program for the following scenario. Use the predicate "{predicate_name_1}" with fields {a} and the predicate "{predicate_name_2}" with fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s7 = f'''Generate an ASP program for this problem. Consider the predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s8 = f'''Compose an ASP program to address the given issue. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s9 = f'''Write an ASP program for this challenge. Consider the predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s10 = f'''Devise an ASP program for the described problem. Use the predicate "{predicate_name_1}" with fields {a} and the predicate "{predicate_name_2}" with fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s11 = f'''Create an ASP program to solve this issue. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s12 = f'''Form an ASP program for the specified problem. Consider the predicate "{predicate_name_1}" with fields {a} and the predicate "{predicate_name_2}" with fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s13 = f'''Write an ASP program to tackle the problem. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s14 = f'''Design an ASP program for this task. Consider the predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s15 = f'''Develop an ASP program to address the issue. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s16 = f'''Compose an ASP program for this problem. Consider the predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s17 = f'''Write an ASP program to solve this problem. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s18 = f'''Draft an ASP program to address the given challenge. Consider the predicate "{predicate_name_1}" with fields {a} and the predicate "{predicate_name_2}" with fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s19 = f'''Generate an ASP program for the described task. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s20 = f'''Formulate an ASP program to address this problem. Consider the predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:

        s1 = f'''Create an ASP script to define the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, given that "{predicate_name_1}" has fields {a} and "{predicate_name_2}" has fields {b}.'''

        s2 = f'''Write an ASP application to address the problem where the predicate "{predicate_name_1}" has fields {a}, and the predicate "{predicate_name_2}" has fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s3 = f'''Develop an ASP solution that defines the predicate "{predicate_name_1}_{predicate_name_2}" to link each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, with "{predicate_name_1}" having fields {a} and "{predicate_name_2}" having fields {b}.'''

        s4 = f'''Compose an ASP code snippet to define the predicate "{predicate_name_1}_{predicate_name_2}" linking each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, using the fields {a} of "{predicate_name_1}" and the fields {b} of "{predicate_name_2}".'''

        s5 = f'''Craft an ASP solution that addresses the problem of defining the predicate "{predicate_name_1}_{predicate_name_2}" which links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, given that "{predicate_name_1}" has fields {a} and "{predicate_name_2}" has fields {b}.'''

        s6 = f'''Generate an ASP program to create the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, with the fields {a} of "{predicate_name_1}" and the fields {b} of "{predicate_name_2}".'''

        s7 = f'''Design an ASP application to solve the problem by defining the predicate "{predicate_name_1}_{predicate_name_2}" which links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, using fields {a} for "{predicate_name_1}" and fields {b} for "{predicate_name_2}".'''

        s8 = f'''Formulate an ASP program that defines the predicate "{predicate_name_1}_{predicate_name_2}" to associate each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, using the fields {a} of "{predicate_name_1}" and {b} of "{predicate_name_2}".'''

        s9 = f'''Compose an ASP script that addresses the problem by defining the predicate "{predicate_name_1}_{predicate_name_2}" which links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}", with "{predicate_name_1}" having fields {a} and "{predicate_name_2}" having fields {b}.'''

        s10 = f'''Create an ASP solution to define the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, given "{predicate_name_1}" has fields {a} and "{predicate_name_2}" has fields {b}.'''

        s11 = f'''Write an ASP program to solve the problem by defining the predicate "{predicate_name_1}_{predicate_name_2}" which associates each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, using the fields {a} of "{predicate_name_1}" and the fields {b} of "{predicate_name_2}".'''

        s12 = f'''Develop an ASP solution to create the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, with "{predicate_name_1}" having fields {a} and "{predicate_name_2}" having fields {b}.'''

        s13 = f'''Draft an ASP script to define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, given "{predicate_name_1}" has fields {a} and "{predicate_name_2}" has fields {b}.'''

        s14 = f'''Generate an ASP program to address the problem of defining the predicate "{predicate_name_1}_{predicate_name_2}" which links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, with "{predicate_name_1}" having fields {a} and "{predicate_name_2}" having fields {b}.'''

        s15 = f'''Craft an ASP solution to define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, using the fields {a} of "{predicate_name_1}" and the fields {b} of "{predicate_name_2}".'''

        s16 = f'''Formulate an ASP program to create the predicate "{predicate_name_1}_{predicate_name_2}" which links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, using fields {a} for "{predicate_name_1}" and fields {b} for "{predicate_name_2}".'''

        s17 = f'''Design an ASP application to solve the problem by defining the predicate "{predicate_name_1}_{predicate_name_2}" which links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, given "{predicate_name_1}" has fields {a} and "{predicate_name_2}" has fields {b}.'''

        s18 = f'''Create an ASP program to define the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, using fields {a} for "{predicate_name_1}" and fields {b} for "{predicate_name_2}".'''

        s19 = f'''Compose an ASP script to address the problem by defining the predicate "{predicate_name_1}_{predicate_name_2}" which associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, with "{predicate_name_1}" having fields {a} and "{predicate_name_2}" having fields {b}.'''

        s20 = f'''Develop an ASP program to solve the problem by creating the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, with "{predicate_name_1}" having fields {a} and "{predicate_name_2}" having fields {b}.'''

        s21 = f'''{incipit()} Consider predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates to each "{predicate_name_1}" the "{random_attribute}" of "{predicate_name_2}".'''

        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s


def execute_join(predicate_name_1, predicate_name_2, attributes, prompt_invariance, test):
    questions, answers = [], []
    rewritten_questions = []

    f = []

    for attributes_1 in range(3, 6):
        for attributes_2 in range(2, 5):

            fact = ''

            n_attributes = attributes_1
            attributes = np.array(attributes, dtype='U18')
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

            question = f'''{incipit()} Consider predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates to each "{predicate_name_1}" the "{random_attribute}" of "{predicate_name_2}".'''

            if (prompt_invariance):
                rewritten_questions = rephrased_execute_join(predicate_name_1, predicate_name_2, a, b, random_attribute,
                                                             test)

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
            rewritten_answers = [answer] * len(rewritten_questions)

            rewritten_facts = np.repeat(fact, len(rewritten_questions))

            if (len(rewritten_questions) > 0):
                questions.extend(rewritten_questions)
                answers.extend(rewritten_answers)
                f.extend(rewritten_facts)
            else:
                questions.append(question)
                answers.append(answer)
                f.append(fact)

    return questions, answers, f


def rephrased_transitive_closure(closure_name, predicate_name, test):
    if not test:
        s1 = f'''Create an ASP program that establishes the transitive closure of the predicate "{predicate_name}", defined as predicate "{closure_name}".'''

        s2 = f'''Write an ASP program that computes the transitive closure of the predicate "{predicate_name}", resulting in the predicate "{closure_name}".'''

        s3 = f'''Develop an ASP solution that derives the predicate "{closure_name}" by extending the transitive closure of the predicate "{predicate_name}".'''

        s4 = f'''Craft an ASP program that constructs the predicate "{closure_name}" based on the transitive closure of the predicate "{predicate_name}".'''

        s5 = f'''Construct an ASP script that defines the predicate "{closure_name}" as the transitive closure of the given predicate "{predicate_name}".'''

        s6 = f'''Generate an ASP code snippet that establishes the transitive closure "{closure_name}" of the predicate "{predicate_name}".'''

        s7 = f'''Produce an ASP implementation that infers the predicate "{closure_name}" using the transitive closure of the predicate"{predicate_name}".'''

        s8 = f'''Write an ASP rule that computes the predicate "{closure_name}" by extending the reachability of the predicate "{predicate_name}".'''

        s9 = f'''Design an ASP program that links the predicate "{closure_name}" to the transitive closure of the predicate "{predicate_name}".'''

        s10 = f'''Compose an ASP solution that defines the predicate "{closure_name}" based on the transitive connections inferred from the predicate "{predicate_name}".'''

        s11 = f'''Create an ASP program that calculates the transitive closure of the predicate "{predicate_name}", defining it as "{closure_name}".'''

        s12 = f'''Write an ASP solution to compute the transitive closure of the predicate "{predicate_name}", resulting in the definition of the predicate "{closure_name}".'''

        s13 = f'''Develop an ASP code that extends the transitive closure of the predicate "{predicate_name}" to form the predicate "{closure_name}".'''

        s14 = f'''Craft an ASP program to construct the predicate "{closure_name}" based on the transitive closure derived from the predicate "{predicate_name}".'''

        s15 = f'''Implement an ASP script that establishes the transitive closure of the predicate "{predicate_name}" and defines it as "{closure_name}".'''

        s16 = f'''Generate an ASP code snippet that links the predicate "{predicate_name}" to its transitive closure, defined as "{closure_name}".'''

        s17 = f'''Produce an ASP program that computes the transitive closure of the predicate "{predicate_name}" and infers the predicate "{closure_name}".'''

        s18 = f'''Write an ASP rule to compute the predicate "{closure_name}" by determining the transitive closure of the predicate "{predicate_name}".'''

        s19 = f'''Design an ASP program to establish the predicate "{closure_name}" through the transitive closure of the predicate "{predicate_name}".'''

        s20 = f'''Compose an ASP solution that links the predicate "{closure_name}" to the transitive closure of the predicate "{predicate_name}".'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:

        s1 = f'''Write an ASP application to compute the transitive closure of the predicate "{predicate_name}", resulting in the definition of the predicate "{closure_name}".'''

        s2 = f'''Compose an ASP solution that calculates the transitive closure of the predicate "{predicate_name}", resulting in the predicate "{closure_name}".'''

        s3 = f'''Develop an ASP script that derives the predicate "{closure_name}" through the transitive closure of the predicate "{predicate_name}".'''

        s4 = f'''Generate an ASP program to construct the predicate "{closure_name}" based on the transitive closure of the predicate "{predicate_name}".'''

        s5 = f'''Create an ASP solution that establishes the transitive closure of the predicate "{predicate_name}", defined as "{closure_name}".'''

        s6 = f'''Formulate an ASP code snippet to establish the predicate "{closure_name}" by computing the transitive closure of the predicate "{predicate_name}".'''

        s7 = f'''Design an ASP program that infers the predicate "{closure_name}" using the transitive closure of the predicate "{predicate_name}".'''

        s8 = f'''Craft an ASP solution to compute the predicate "{closure_name}" by extending the transitive closure of the predicate "{predicate_name}".'''

        s9 = f'''Produce an ASP script that links the predicate "{closure_name}" to the transitive closure of the predicate "{predicate_name}".'''

        s10 = f'''Write an ASP application that defines the predicate "{closure_name}" based on the transitive closure of the predicate "{predicate_name}".'''

        s11 = f'''Create an ASP code snippet to determine the transitive closure of the predicate "{predicate_name}", resulting in the predicate "{closure_name}".'''

        s12 = f'''Generate an ASP solution that computes the transitive closure of the predicate "{predicate_name}", defining the predicate "{closure_name}".'''

        s13 = f'''Compose an ASP script to extend the transitive closure of the predicate "{predicate_name}" and form the "{closure_name}".'''

        s14 = f'''Develop an ASP application that constructs the predicate "{closure_name}" based on the transitive closure of the predicate "{predicate_name}".'''

        s15 = f'''Formulate an ASP solution to establish the transitive closure of the predicate "{predicate_name}", defined as "{closure_name}".'''

        s16 = f'''Design an ASP code to link the predicate "{predicate_name}" to its transitive closure, defined as "{closure_name}".'''

        s17 = f'''Craft an ASP script that infers the predicate "{closure_name}" by computing the transitive closure of the predicate "{predicate_name}".'''

        s18 = f'''Produce an ASP program to compute the transitive closure of the predicate "{predicate_name}" and define it as "{closure_name}".'''

        s19 = f'''Create an ASP solution that establishes the predicate "{closure_name}" through the transitive closure of the predicate "{predicate_name}".'''

        s20 = f'''Develop an ASP script to link the predicate "{predicate_name}" to its transitive closure, resulting in the predicate "{closure_name}".'''

        s21 = f'''{incipit()} Define predicate "{closure_name}" as the transitive closure of predicate "{predicate_name}".'''

        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s


def transitive_closure(closure_name, predicate_name, prompt_invariance, test):
    questions, answers = [], []
    rewritten_questions = []

    question = f'''{incipit()} Define predicate "{closure_name}" as the transitive closure of predicate "{predicate_name}".'''

    if (prompt_invariance):
        rewritten_questions = rephrased_transitive_closure(closure_name, predicate_name, test)

    answer = f'''{closure_name}(X,Y):-{predicate_name}(X,Y).\n{closure_name}(X,Y):-{predicate_name}(X,Z),{closure_name}(Z,Y).'''
    rewritten_answers = [answer] * len(rewritten_questions)

    if (len(rewritten_questions) > 0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    f = f'''{predicate_name}(1..3, 1..4).'''

    return questions, answers, f


def rephrased_preferences(predicate_name, label, value, cost_value, cost_level, test):
    if not test:
        s1 = f'''Create an ASP program to preferably ensure that the predicate "{predicate_name}" with value "{value}" is not associated with "{label}". If this association occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s2 = f'''Write an ASP program which tries to avoid linking the predicate "{predicate_name}" with value "{value}" to "{label}". If such a link is found, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s3 = f'''Develop an ASP solution to keep the predicate "{predicate_name}" with value "{value}" preferably separate from "{label}". If this association occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s4 = f'''Craft an ASP program to discourage the predicate "{predicate_name}" with value "{value}" being associate with "{label}". If it does, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s5 = f'''Formulate an ASP script to minimize occurrences where the predicate "{predicate_name}" with value "{value}" is linked with "{label}". If such a connection happens, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s6 = f'''Generate an ASP code which tries to ensure that the predicate "{predicate_name}" with value "{value}" remains unlinked to "{label}". Any occurrence of this link incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s7 = f'''Create an ASP implementation to discourage the predicate "{predicate_name}" with value "{value}" being associated with "{label}". If it is, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s8 = f'''Write an ASP rule which prefers that the predicate "{predicate_name}" with value "{value}" is not linked to "{label}". If such an association exists, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s9 = f'''Design an ASP program to keep the predicate "{predicate_name}" with value "{value}" preferably unlinked from "{label}". If they are associated, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s10 = f'''Compose an ASP solution which tries to enforce the predicate "{predicate_name}" with value "{value}" to be not linked to "{label}". If this link is established, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s11 = f'''Formulate an ASP program to preferably ensure the predicate "{predicate_name}" with value "{value}" does not match with "{label}". If this match occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s12 = f'''Create an ASP script that tries to keep the predicate "{predicate_name}" with value "{value}" separate from "{label}". If such a connection is found, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s13 = f'''Develop an ASP solution to minimize occurrences where the predicate "{predicate_name}" with value "{value}" is connected with "{label}". If this connection exists, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s14 = f'''Craft an ASP program to discourage the predicate "{predicate_name}" with value "{value}" to be associated with "{label}". If it does, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s15 = f'''Generate an ASP code snippet to try ensuring the predicate "{predicate_name}" with value "{value}" is not linked to "{label}". If this association is found, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s16 = f'''Compose an ASP solution to try avoiding the predicate "{predicate_name}" with value "{value}" being linked to "{label}". If such an association exists, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s17 = f'''Build an ASP program which preferably keeps the predicate "{predicate_name}" with value "{value}" separate from "{label}". If they are connected, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s18 = f'''Develop an ASP script to discourage the predicate "{predicate_name}" with value "{value}" being linked to "{label}". If this link is found, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s19 = f'''Create an ASP program which tries to avoid the predicate "{predicate_name}" with value "{value}" is linked to "{label}". If this link occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s20 = f'''Formulate an ASP solution to try ensuring the predicate "{predicate_name}" with value "{value}" is not connected to "{label}". If this connection happens, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:

        s1 = f'''Design an ASP solution to preferably prevent the predicate "{predicate_name}" with value "{value}" from being linked to "{label}". If this occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s2 = f'''Craft an ASP program which tries to ensure that the predicate "{predicate_name}" with value "{value}" is not associated with "{label}", incurring a cost of "{cost_value}" at level "{cost_level}" if it does.'''

        s3 = f'''Develop an ASP code snippet to minimize linking the predicate "{predicate_name}" with value "{value}" to "{label}". If such a link is found, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s4 = f'''Write an ASP program that preferably disallows the association between "{predicate_name}" with value "{value}" and "{label}", with a cost of "{cost_value}" at level "{cost_level}" if this association occurs.'''

        s5 = f'''Generate an ASP application to discourage the predicate "{predicate_name}" with value "{value}" be linked to "{label}", incurring a cost of "{cost_value}" at level "{cost_level}" if associated.'''

        s6 = f'''Compose an ASP script to preferably ensure the predicate "{predicate_name}" with value "{value}" does not link to "{label}". If this connection happens, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s7 = f'''Formulate an ASP solution which tries to prevent the association of the predicate "{predicate_name}" with value "{value}" with "{label}". If this association occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s8 = f'''Create an ASP program that minimizes occurrences where the predicate "{predicate_name}" with value "{value}" is linked to "{label}". If linked, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s9 = f'''Develop an ASP application to try avoiding the predicate "{predicate_name}" with value "{value}" being associated with "{label}", incurring a cost of "{cost_value}" at level "{cost_level}" if found.'''

        s10 = f'''Generate an ASP script to preferably ensure the predicate "{predicate_name}" with value "{value}" is not linked to "{label}". Any occurrence incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s11 = f'''Draft an ASP solution which tries to make sure the predicate "{predicate_name}" with value "{value}" is not connected to "{label}". If connected, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s12 = f'''Write an ASP application that tries avoiding the predicate "{predicate_name}" with value "{value}" from being linked to "{label}", incurring a cost of "{cost_value}" at level "{cost_level}" if linked.'''

        s13 = f'''Compose an ASP program to discourage the predicate "{predicate_name}" with value "{value}" being associated to "{label}". If this association occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s14 = f'''Craft an ASP solution to discourage the linking of the predicate "{predicate_name}" with value "{value}" to "{label}". Any link incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s15 = f'''Create an ASP code to tries to ensure that the predicate "{predicate_name}" with value "{value}" does not associate with "{label}". If it does, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s16 = f'''Formulate an ASP application to minimize occurrences where the predicate "{predicate_name}" with value "{value}" is linked to "{label}". If linked, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s17 = f'''Generate an ASP program to discourage the association of the predicate "{predicate_name}" with value "{value}" with "{label}". If associated, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s18 = f'''Develop an ASP script which tries to keep the predicate "{predicate_name}" with value "{value}" unlinked from "{label}". Any occurrence incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s19 = f'''Compose an ASP solution to discourage the linking of the predicate "{predicate_name}" with value "{value}" to "{label}". Any link incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s20 = f'''Craft an ASP application to mimize the situations where the predicate "{predicate_name}" with value "{value}" is associated with "{label}". If this occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s21 = f'''{incipit()} I would prefer that predicate "{predicate_name}" with value "{value}" is not associated with "{label}". If this occurs, it costs "{cost_value}" at level "{cost_level}".'''

        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s


def preferences(predicate_name, labels, prompt_invariance, test):
    questions, answers, f = [], [], []
    rewritten_questions = []
    n_values = 20

    for cost_value in range(1, 3):
        for cost_level in range(1, 3):
            value = np.random.randint(1, n_values)

            label = labels[np.random.randint(0, len(labels))]
            question = f'''{incipit()} I would prefer that predicate "{predicate_name}" with value "{value}" is not associated with "{label}". If this occurs, it costs "{cost_value}" at level "{cost_level}".'''

            if (prompt_invariance):
                rewritten_questions = rephrased_preferences(predicate_name, label, value, cost_value, cost_level, test)

            answer = f''':~assign({value},"{label}").[{cost_value}@{cost_level}]'''
            rewritten_answers = [answer] * len(rewritten_questions)

            fact = f'''{predicate_name}(1..{n_values}).'''

            for label in labels[:-1]:
                fact += f'''assign(X,"{label}")|'''
            fact += f'''assign(X,"{labels[-1]}"):-{predicate_name}(X).'''

            if (len(rewritten_questions) > 0):
                questions.extend(rewritten_questions)
                answers.extend(rewritten_answers)
            else:
                questions.append(question)
                answers.append(answer)

    return questions, answers, fact


def rephrased_minimizing(predicate_name, label, test):
    if not test:
        s1 = f'''Write an ASP program to minimize the number of predicate "{predicate_name}" tagged with label "{label}".'''

        s2 = f'''Create an ASP program to reduce the number of predicate "{predicate_name}" associated with label "{label}".'''

        s3 = f'''Compose an ASP program to limit the quantity of predicate "{predicate_name}" labeled with "{label}".'''

        s4 = f'''Draft an ASP program to decrease the instances of predicate "{predicate_name}" tagged as "{label}".'''

        s5 = f'''Generate an ASP program to cut down on the number of predicate "{predicate_name}" marked with label "{label}".'''

        s6 = f'''Produce an ASP program to lower the count of predicate "{predicate_name}" connected to label "{label}".'''

        s7 = f'''Write an ASP program to lessen the amount of predicate "{predicate_name}" designated with label "{label}".'''

        s8 = f'''Create an ASP program to diminish the frequency of predicate "{predicate_name}" tagged with "{label}".'''

        s9 = f'''Design an ASP program to shrink the number of predicate "{predicate_name}" that has the label "{label}".'''

        s10 = f'''Formulate an ASP program to reduce the occurrence of predicate "{predicate_name}" associated with label "{label}".'''

        s11 = f'''Develop an ASP script to minimize the number of "{predicate_name}" predicates labeled as "{label}".'''

        s12 = f'''Create an ASP solution to reduce the frequency of "{predicate_name}" predicates tagged with "{label}".'''

        s13 = f'''Write an ASP code snippet to limit the instances of "{predicate_name}" predicates associated with "{label}".'''

        s14 = f'''Compose an ASP application to cut down the quantity of "{predicate_name}" predicates labeled "{label}".'''

        s15 = f'''Draft an ASP solution to decrease the number of "{predicate_name}" predicates marked with label "{label}".'''

        s16 = f'''Generate an ASP script to lower the count of "{predicate_name}" predicates connected to label "{label}".'''

        s17 = f'''Produce an ASP application to lessen the amount of "{predicate_name}" predicates designated with "{label}".'''

        s18 = f'''Write an ASP program to diminish the frequency of "{predicate_name}" predicates tagged as "{label}".'''

        s19 = f'''Create an ASP script to shrink the number of "{predicate_name}" predicates that have the label "{label}".'''

        s20 = f'''Formulate an ASP solution to reduce the occurrence of "{predicate_name}" predicates associated with label "{label}".'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:

        s1 = f'''Write an ASP program to minimize the number of predicate "{predicate_name}" tagged with label "{label}".'''

        s2 = f'''Create an ASP program to reduce the number of predicate "{predicate_name}" associated with label "{label}".'''

        s3 = f'''Compose an ASP program to limit the quantity of predicate "{predicate_name}" labeled with "{label}".'''

        s4 = f'''Draft an ASP program to decrease the instances of predicate "{predicate_name}" tagged as "{label}".'''

        s5 = f'''Generate an ASP program to cut down on the number of predicate "{predicate_name}" marked with label "{label}".'''

        s6 = f'''Produce an ASP program to lower the count of predicate "{predicate_name}" connected to label "{label}".'''

        s7 = f'''Write an ASP program to lessen the amount of predicate "{predicate_name}" designated with label "{label}".'''

        s8 = f'''Create an ASP program to diminish the frequency of predicate "{predicate_name}" tagged with "{label}".'''

        s9 = f'''Design an ASP program to shrink the number of predicate "{predicate_name}" that has the label "{label}".'''

        s10 = f'''Formulate an ASP program to reduce the occurrence of predicate "{predicate_name}" associated with label "{label}".'''

        s11 = f'''Develop an ASP script to minimize the number of "{predicate_name}" predicates labeled as "{label}".'''

        s12 = f'''Create an ASP solution to reduce the frequency of "{predicate_name}" predicates tagged with "{label}".'''

        s13 = f'''Write an ASP code snippet to limit the instances of "{predicate_name}" predicates associated with "{label}".'''

        s14 = f'''Compose an ASP application to cut down the quantity of "{predicate_name}" predicates labeled "{label}".'''

        s15 = f'''Draft an ASP solution to decrease the number of "{predicate_name}" predicates marked with label "{label}".'''

        s16 = f'''Generate an ASP script to lower the count of "{predicate_name}" predicates connected to label "{label}".'''

        s17 = f'''Produce an ASP application to lessen the amount of "{predicate_name}" predicates designated with "{label}".'''

        s18 = f'''Write an ASP program to diminish the frequency of "{predicate_name}" predicates tagged as "{label}".'''

        s19 = f'''Create an ASP script to shrink the number of "{predicate_name}" predicates that have the label "{label}".'''

        s20 = f'''Formulate an ASP solution to reduce the occurrence of "{predicate_name}" predicates associated with label "{label}".'''

        s = []

        for i in range(1, 21):
            s.append(vars()['s' + str(i)])

    return s


def minimizing(predicate_name, labels, prompt_invariance, test):
    questions, answers = [], []
    rewritten_questions = []

    label = labels[np.random.randint(0, len(labels))]

    question = f'''{incipit()} Minimize the number of predicate "{predicate_name}" tagged with label "{label}".'''

    if (prompt_invariance):
        rewritten_questions = rephrased_minimizing(predicate_name, label, test)

    answer = f''':~{predicate_name}(X),assign(X,"{label}").[1@1,X]'''
    rewritten_answers = [answer] * len(rewritten_questions)

    if (len(rewritten_questions) > 0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    # USELESS FOR NOW  
    f = []

    return questions, answers, f


def rephrased_maximizing(predicate_name, label, test):
    if not test:
        s1 = f'''Write an ASP program to maximize the occurrence of "{predicate_name}" predicates labeled as "{label}".'''

        s2 = f'''Create an ASP program to increase the frequency of "{predicate_name}" predicates with the label "{label}".'''

        s3 = f'''Compose an ASP program to boost the number of "{predicate_name}" predicates tagged with "{label}".'''

        s4 = f'''Draft an ASP program to enhance the occurrence of "{predicate_name}" predicates identified with the label "{label}".'''

        s5 = f'''Generate an ASP program to amplify the instances of "{predicate_name}" predicates marked as "{label}".'''

        s6 = f'''Produce an ASP program to elevate the count of "{predicate_name}" predicates associated with "{label}".'''

        s7 = f'''Write an ASP program to heighten the occurrence of "{predicate_name}" predicates labeled "{label}".'''

        s8 = f'''Create an ASP program to raise the number of "{predicate_name}" predicates tagged with "{label}".'''

        s9 = f'''Design an ASP program to increase the presence of "{predicate_name}" predicates with the label "{label}".'''

        s10 = f'''Formulate an ASP program to maximize the frequency of "{predicate_name}" predicates identified with "{label}".'''

        s11 = f'''Develop an ASP script to maximize the occurrence of "{predicate_name}" predicates labeled as "{label}".'''

        s12 = f'''Create an ASP solution to increase the number of "{predicate_name}" predicates with the label "{label}".'''

        s13 = f'''Write an ASP code snippet to boost the instances of "{predicate_name}" predicates tagged with "{label}".'''

        s14 = f'''Compose an ASP application to enhance the number of "{predicate_name}" predicates identified with the label "{label}".'''

        s15 = f'''Draft an ASP solution to amplify the frequency of "{predicate_name}" predicates marked as "{label}".'''

        s16 = f'''Generate an ASP script to elevate the count of "{predicate_name}" predicates associated with "{label}".'''

        s17 = f'''Produce an ASP application to heighten the occurrence of "{predicate_name}" predicates labeled as "{label}".'''

        s18 = f'''Write an ASP program to raise the number of "{predicate_name}" predicates tagged with "{label}".'''

        s19 = f'''Create an ASP script to increase the presence of "{predicate_name}" predicates with the label "{label}".'''

        s20 = f'''Formulate an ASP solution to maximize the instances of "{predicate_name}" predicates identified with "{label}".'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:

        s1 = f'''Write an ASP program to maximize the occurrence of "{predicate_name}" predicates labeled as "{label}".'''

        s2 = f'''Create an ASP program to increase the frequency of "{predicate_name}" predicates with the label "{label}".'''

        s3 = f'''Compose an ASP program to boost the number of "{predicate_name}" predicates tagged with "{label}".'''

        s4 = f'''Draft an ASP program to enhance the occurrence of "{predicate_name}" predicates identified with the label "{label}".'''

        s5 = f'''Generate an ASP program to amplify the instances of "{predicate_name}" predicates marked as "{label}".'''

        s6 = f'''Produce an ASP program to elevate the count of "{predicate_name}" predicates associated with "{label}".'''

        s7 = f'''Write an ASP program to heighten the occurrence of "{predicate_name}" predicates labeled "{label}".'''

        s8 = f'''Create an ASP program to raise the number of "{predicate_name}" predicates tagged with "{label}".'''

        s9 = f'''Design an ASP program to increase the presence of "{predicate_name}" predicates with the label "{label}".'''

        s10 = f'''Formulate an ASP program to maximize the frequency of "{predicate_name}" predicates identified with "{label}".'''

        s11 = f'''Develop an ASP script to maximize the occurrence of "{predicate_name}" predicates labeled as "{label}".'''

        s12 = f'''Create an ASP solution to increase the number of "{predicate_name}" predicates with the label "{label}".'''

        s13 = f'''Write an ASP code snippet to boost the instances of "{predicate_name}" predicates tagged with "{label}".'''

        s14 = f'''Compose an ASP application to enhance the number of "{predicate_name}" predicates identified with the label "{label}".'''

        s15 = f'''Draft an ASP solution to amplify the frequency of "{predicate_name}" predicates marked as "{label}".'''

        s16 = f'''Generate an ASP script to elevate the count of "{predicate_name}" predicates associated with "{label}".'''

        s17 = f'''Produce an ASP application to heighten the occurrence of "{predicate_name}" predicates labeled as "{label}".'''

        s18 = f'''Write an ASP program to raise the number of "{predicate_name}" predicates tagged with "{label}".'''

        s19 = f'''Create an ASP script to increase the presence of "{predicate_name}" predicates with the label "{label}".'''

        s20 = f'''Formulate an ASP solution to maximize the instances of "{predicate_name}" predicates identified with "{label}".'''

        s = []

        for i in range(1, 21):
            s.append(vars()['s' + str(i)])

    return s


def maximizing(predicate_name, labels, prompt_invariance, test):
    questions, answers = [], []
    rewritten_questions = []

    label = labels[np.random.randint(0, len(labels))]

    question = f'''{incipit()} Maximize the occurrence of "{predicate_name}" predicates labeled as "{label}".'''

    if (prompt_invariance):
        rewritten_questions = rephrased_maximizing(predicate_name, label, test)

    answer = f''':~{predicate_name}(X),not assign(X,"{label}").[1@1,X]'''
    rewritten_answers = [answer] * len(rewritten_questions)

    if (len(rewritten_questions) > 0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    f = []

    return questions, answers, f


def rephrased_select_by_negative_condition(predicate_name, not_predicate_name, label, test):
    if not test:
        s1 = f'''Write an ASP program to select all values associated with the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and label "{label}".'''

        s2 = f'''Create an ASP program to fetch all values linked to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s3 = f'''Compose an ASP program to identify all values connected to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and the label "{label}".'''

        s4 = f'''Draft an ASP program to select all values tied to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and having the label "{label}".'''

        s5 = f'''Generate an ASP program to retrieve all values associated with the predicate "{predicate_name}" but not with the predicate "{not_predicate_name}" and labeled "{label}".'''

        s6 = f'''Produce an ASP program to gather all values connected to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and the label "{label}".'''

        s7 = f'''Write an ASP script to select all values linked to the predicate "{predicate_name}" but not tied to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s8 = f'''Create an ASP program to choose all values associated with the predicate "{predicate_name}" but not with the predicate "{not_predicate_name}" and the label "{label}".'''

        s9 = f'''Design an ASP program to find all values associated with the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and carrying the label "{label}".'''

        s10 = f'''Formulate an ASP program to gather all values tied to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and labeled "{label}".'''

        s11 = f'''Develop an ASP script to collect all values related to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and the label "{label}".'''

        s12 = f'''Create an ASP solution to identify all values linked to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s13 = f'''Compose an ASP code to find all values connected to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and carrying the label "{label}".'''

        s14 = f'''Draft an ASP application to select all values tied to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s15 = f'''Generate an ASP solution to retrieve all values associated with the predicate "{predicate_name}" but not with the predicate "{not_predicate_name}" and having the label "{label}".'''

        s16 = f'''Write an ASP program to fetch all values linked to the predicate "{predicate_name}" but not tied to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s17 = f'''Create an ASP script to gather all values associated with the predicate "{predicate_name}" but not connected to the predicate "{not_predicate_name}" and the label "{label}".'''

        s18 = f'''Design an ASP program to capture all values related to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s19 = f'''Compose an ASP code snippet to identify all values connected to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s20 = f'''Generate an ASP script to select all values tied to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and having the label "{label}".'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:

        s1 = f'''Write an ASP script to select all values tied to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled as "{label}".'''

        s2 = f'''Create an ASP application to fetch values associated with the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s3 = f'''Compose an ASP solution to identify all values connected to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s4 = f'''Draft an ASP program to retrieve values tied to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and labeled "{label}".'''

        s5 = f'''Generate an ASP script to gather values linked to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s6 = f'''Produce an ASP code snippet to collect values associated with the predicate "{predicate_name}" but not connected to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s7 = f'''Write an ASP application to select values tied to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s8 = f'''Create an ASP solution to fetch values connected to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and labeled "{label}".'''

        s9 = f'''Design an ASP program to identify values linked to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s10 = f'''Formulate an ASP code to gather values associated with the predicate "{predicate_name}" but not connected to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s11 = f'''Develop an ASP script to collect values tied to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s12 = f'''Create an ASP program to capture values associated with the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s13 = f'''Compose an ASP application to find values connected to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s14 = f'''Draft an ASP solution to identify values associated with the predicate "{predicate_name}" but not tied to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s15 = f'''Generate an ASP code snippet to retrieve values linked to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s16 = f'''Produce an ASP program to gather values associated with the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s17 = f'''Write an ASP script to select values connected to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and labeled "{label}".'''

        s18 = f'''Create an ASP application to collect values tied to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s19 = f'''Design an ASP solution to capture values associated with the predicate "{predicate_name}" but not tied to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s20 = f'''Formulate an ASP code to select values linked to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and labeled "{label}".'''

        s21 = f'''{incipit()} Select all values associated with predicate "{predicate_name}" but not associated with predicate "{not_predicate_name}" and label "{label}".'''

        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s


def select_by_negative_condition(predicate_name, not_predicate_name, labels, prompt_invariance, test):
    questions, answers = [], []
    rewritten_questions = []

    label = labels[np.random.randint(0, len(labels))]

    question = f'''{incipit()} Select all values associated with predicate "{predicate_name}" but not associated with predicate "{not_predicate_name}" and label "{label}".'''

    if (prompt_invariance):
        rewritten_questions = rephrased_select_by_negative_condition(predicate_name, not_predicate_name, label, test)

    answer = f'''select(X):-{predicate_name}(X),not {not_predicate_name}(X,"{label}").'''
    rewritten_answers = [answer] * len(rewritten_questions)

    chosen_labels = list(set(list(np.random.choice(labels, size=4, replace=False))).union({label}))
    combinations = list(zip(range(1, 4), chosen_labels))

    if (len(rewritten_questions) > 0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    fact = f'''{predicate_name}(1..3).'''

    for i, l in combinations:
        fact += f'''{not_predicate_name}({i},"{l}").'''

    f = fact

    return questions, answers, f


def rephrased_select_by_numeric_condition(predicate_name, condition, condition_value, test):
    if not test:
        s1 = f'''Write an ASP program to select all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s2 = f'''Create an ASP program to fetch all values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s3 = f'''Compose an ASP program to identify all values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s4 = f'''Draft an ASP program to select all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s5 = f'''Generate an ASP program to retrieve all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s6 = f'''Produce an ASP program to gather all values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s7 = f'''Write an ASP script to select all values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s8 = f'''Create an ASP program to choose all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s9 = f'''Design an ASP program to find all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s10 = f'''Formulate an ASP program to gather all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s11 = f'''Develop an ASP script to select all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s12 = f'''Create an ASP solution to fetch all values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s13 = f'''Compose an ASP code to identify all values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s14 = f'''Draft an ASP application to select all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s15 = f'''Generate an ASP solution to retrieve all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s16 = f'''Write an ASP program to fetch all values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s17 = f'''Create an ASP script to gather all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s18 = f'''Design an ASP program to capture all values related to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s19 = f'''Compose an ASP code snippet to identify all values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s20 = f'''Generate an ASP script to select all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:

        s1 = f'''Create an ASP application to fetch all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s2 = f'''Write an ASP solution to select values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s3 = f'''Develop an ASP program to gather all values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s4 = f'''Formulate an ASP script to identify values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s5 = f'''Craft an ASP code to retrieve values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s6 = f'''Generate an ASP application to select all values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s7 = f'''Compose an ASP program to fetch values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s8 = f'''Design an ASP solution to capture all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s9 = f'''Draft an ASP code snippet to identify values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s10 = f'''Produce an ASP script to retrieve values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s11 = f'''Create an ASP application to select values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s12 = f'''Formulate an ASP solution to gather all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s13 = f'''Craft an ASP program to fetch values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s14 = f'''Generate an ASP code to capture values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s15 = f'''Develop an ASP application to retrieve all values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s16 = f'''Compose an ASP script to select values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s17 = f'''Write an ASP solution to identify values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s18 = f'''Design an ASP program to gather values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s19 = f'''Formulate an ASP application to fetch values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s20 = f'''Craft an ASP code snippet to select values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s21 = f'''{incipit()} Select all values associated with predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s


def select_by_numeric_condition(predicate_name, prompt_invariance, test):
    # condition \in [!=, <, >, <=, >=]

    n_values = 100

    condition_dict = {"different": "!=", "greater": ">", "lower": "<", "greater or equal": ">=", "lower or equal": "<="}

    questions, answers = [], []
    rewritten_questions = []

    for condition, condition_symbol in condition_dict.items():
        condition_value = np.random.randint(1, n_values)

        question = f'''{incipit()} Select all values associated with predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        if (prompt_invariance):
            rewritten_questions = rephrased_select_by_numeric_condition(predicate_name, condition, condition_value,
                                                                        test)

        answer = f'''select(X):-{predicate_name}(X,C),C{condition_symbol}{condition_value}.'''
        rewritten_answers = [answer] * len(rewritten_questions)

        if (len(rewritten_questions) > 0):
            questions.extend(rewritten_questions)
            answers.extend(rewritten_answers)
        else:
            questions.append(question)
            answers.append(answer)

    f = f'''{predicate_name}(1..3, 1..{n_values}).'''

    return questions, answers, f



def join_filtering(predicate_name_1, predicate_name_2, attributes, predicates):
    questions, answers = [], []
    rewritten_questions = []

    f = []

    for attributes_1 in range(3, 6):
        for attributes_2 in range(2, 5):

            fact = ''

            n_attributes = attributes_1
            attributes = np.array(attributes, dtype='U18')
            chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
            random_pos = np.random.randint(1, n_attributes)
            chosen_attributes[0] = f"ID"
            chosen_attributes[random_pos] = f"{predicate_name_2}ID"

            string_chosen_attributes = f'''{''.join([f'"{x}",' for x in chosen_attributes[:-1]])}'''
            string_chosen_attributes += f'"{chosen_attributes[-1]}"'

            chosen_labels = np.random.choice(predicates, size=n_attributes, replace=False)

            if (random_pos == 1):
                string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:random_pos - 1]])}'''
                string_chosen_labels += f'"{chosen_labels[-(random_pos - 1)]}"'
                fact += f'''{predicate_name_1}(0..3, 0..4,{string_chosen_labels}).'''

            elif (random_pos > 1):
                if (random_pos < n_attributes - 1):
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:random_pos - 1]])}'''
                    string_chosen_labels += f'"{chosen_labels[-(random_pos - 1)]}"'
                    fact += f'''{predicate_name_1}(0..3,{string_chosen_labels},'''
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[random_pos + 1:-1]])}'''
                    string_chosen_labels += f'"{chosen_labels[-1]}"'
                    fact += f'''0..4, {string_chosen_labels}).'''
                elif (random_pos == n_attributes - 1):
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:-1]])}'''
                    fact += f'''{predicate_name_1}(0..3,{string_chosen_labels}0..4).'''

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
            random_pos_2 = np.random.randint(1, attributes_2)
            chosen_attributes[0] = "ID"

            string_chosen_attributes_2 = f'''{''.join([f'"{x}",' for x in chosen_attributes[:-1]])}'''
            string_chosen_attributes_2 += f'"{chosen_attributes[-1]}"'

            chosen_labels = np.random.choice(predicates, size=n_attributes + 1, replace=False)
            not_label = chosen_labels[-1]

            if (random_pos_2 == 1):
                if (n_attributes == 2):
                    fact += f'''{predicate_name_2}(0..2,"{chosen_labels[1]}").'''
                    fact += f'''{predicate_name_2}(2..4,"{not_label}").'''
                else:
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:-2]])}'''
                    string_chosen_labels += f'"{chosen_labels[-2]}"'
                    fact += f'''{predicate_name_2}(0..2,{string_chosen_labels}).'''
                    chosen_labels[random_pos_2] = not_label
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:-2]])}'''
                    string_chosen_labels += f'"{chosen_labels[-2]}"'
                    fact += f'''{predicate_name_2}(2..4,{string_chosen_labels}).'''

            elif (random_pos_2 > 1):
                if (random_pos_2 < n_attributes - 1):
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:random_pos_2 - 1]])}'''
                    string_chosen_labels += f'"{chosen_labels[-(random_pos_2 - 1)]}"'
                    fact += f'''{predicate_name_2}(0..3,{string_chosen_labels}).'''
                    chosen_labels[random_pos_2] = not_label
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:random_pos_2 - 1]])}'''
                    string_chosen_labels += f'"{chosen_labels[-(random_pos_2 - 1)]}"'
                    fact += f'''{predicate_name_2}(2..4,{string_chosen_labels}).'''
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[random_pos_2 + 1:-2]])}'''
                    string_chosen_labels += f'"{chosen_labels[-2]}"'
                    fact += f'''0..4, {string_chosen_labels}).'''
                    chosen_labels[random_pos_2] = not_label
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[random_pos_2 + 1:-2]])}'''
                    string_chosen_labels += f'"{chosen_labels[-2]}"'
                    fact += f'''{predicate_name_2}(2..4,{string_chosen_labels}).'''
                elif (random_pos_2 == n_attributes - 1):
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:-2]])}'''
                    string_chosen_labels += f'"{chosen_labels[-2]}"'
                    fact += f'''{predicate_name_2}(0..2,{string_chosen_labels}).'''
                    chosen_labels[random_pos_2] = not_label
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:-2]])}'''
                    string_chosen_labels += f'"{chosen_labels[-2]}"'
                    fact += f'''{predicate_name_2}(2..4,{string_chosen_labels}).'''

            random_attribute_index = random_pos_2
            random_attribute = chosen_attributes[random_attribute_index]

            b = ''
            for attr in chosen_attributes[:-1]:
                b += f'"{attr}",'
            b += f'"{chosen_attributes[-1]}"'

            question = f'''{incipit()} Consider predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates to each {predicate_name_1} the attribute {random_attribute} of {predicate_name_2}. In addition, select all values associated to the predicate "{predicate_name_1}_{predicate_name_2}" with label "{not_label}".'''

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

            answer = f'''{predicate_name_1}_{predicate_name_2}(X,Z):-{p},{q}.\nselect(X):-{predicate_name_1}_{predicate_name_2}(X,"{not_label}").'''
            rewritten_answers = [answer] * len(rewritten_questions)

            rewritten_facts = np.repeat(fact, len(rewritten_questions))

            if (len(rewritten_questions) > 0):
                questions.extend(rewritten_questions)
                answers.extend(rewritten_answers)
                f.extend(rewritten_facts)
            else:
                questions.append(question)
                answers.append(answer)
                f.append(fact)

    return questions, answers, f


def guessing_constraint(labels, predicate_name):
    f = []

    questions, answers = [], []
    rewritten_questions = []

    n_max = 10

    n_values = 20

    value = np.random.randint(1, n_values)

    n_labels = np.random.randint(2, n_max)
    labels_to_assign = np.random.choice(labels, size=n_labels,
                                        replace=False)  # dalla raccolta di etichette ne sceglie "size" e senza rimpiazzo   Crea quindi un array di dimensione size scegliendo casualmente gli elementi da "labels"
    notlabel = np.random.choice(labels_to_assign)
    question = f'''{incipit()} Assign exactly a label among a given set of labels to a set of elements. The set of elements is expressed by predicate {predicate_name}. The labels are {','.join([f"{x}" for x in labels_to_assign])}. Then prevent the predicate "{predicate_name}" with value "{value}" from having label "{notlabel}".'''

    answer = ""
    for label in labels_to_assign[:-1]:
        answer += f'''assign(X,"{label}")|'''
    answer += f'''assign(X,"{labels_to_assign[-1]}"):-{predicate_name}(X).\n:-assign({value}, "{notlabel}").'''

    rewritten_answers = [answer] * len(rewritten_questions)

    if (len(rewritten_questions) > 0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    f.append(f"{predicate_name}(1..{n_values}).")

    return questions, answers, f


def combination_negative_filtering(labels, predicate_name_1, predicate_name_2, predicate_name_3):

    f = []
    questions, answers = [], []
    rewritten_questions = []

    some_labels = np.random.choice(labels, size=3, replace=False)
    label = some_labels[-1]

    question = f'''{incipit()} Generate all the combinations of elements from two sets. The two sets are represented by predicates "{predicate_name_1}" and "{predicate_name_2}". In addition, select all values associated with predicate combination but not associated with predicate "{predicate_name_3}" and label "{label}".'''

    answer = f'''combination(X,Y):-{predicate_name_1}(X),{predicate_name_2}(Y).\nselect(X):-combination(X,_), not {predicate_name_3}(X, "{label}").'''
    rewritten_answers = [answer] * len(rewritten_questions)

    fact = f'''{predicate_name_1}(1..4).{predicate_name_2}(1..5).{predicate_name_3}(0..1,"{label}").'''
    for l in some_labels[:-1]:
        fact += f'''{predicate_name_3}(2..3,"{l}").'''

    if (len(rewritten_questions) > 0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
        f.extend(fact)
    else:
        questions.append(question)
        answers.append(answer)
        f.append(fact)

    return questions, answers, f


colors = ["red", "green", "blue", "yellow", "brown", "orange", "purple", "gray", "cyan"]
cities = ["rome", "paris", "venice", "new york", "london", "amsterdam", "dubai", "tokyo", "shangai", "florence"]
labels = ["color", "person", "tree", "car", "moto", "bike", "table", "food", "element", "street", "object"]
attributes = ["price", "name", "city", "age", "author", "creator", "shape", "height", "description"]

predicates = colors + cities + labels + attributes
closures = ["path", "flights", "ancestors", "destinations", "arrivals"]


def generate_subproblems(turn, size, train_size, validation, print_proportions=False):
    questions = []
    answers = []
    facts = []

    data_folder = "data/"

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    for i in tqdm(range(size), total=size):
        if not validation:
            np.random.seed(i)
        else:
            np.random.seed(train_size + i)

        match turn:
            case "core":

                for _ in range(10):
                    question_assignments, answer_assignments, f = label_assignment(predicates,
                                                                                   np.random.choice(predicates), False,
                                                                                   False)
                    questions.extend(question_assignments)
                    answers.extend(answer_assignments)
                    facts.extend(f)

                n_questions_assignment = len(questions)

                for _ in range(5):
                    question_prevents, answer_prevents, f = prevent_value(predicates, np.random.choice(predicates),
                                                                          False, False)
                    questions.extend(question_prevents)
                    answers.extend(answer_prevents)
                    facts.extend(f)

                n_questions_prevent = len(questions) - n_questions_assignment

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_combinations, answers_combinations, f = generate_combinations(p_1, p_2, False, False)

                questions.extend(questions_combinations)
                answers.extend(answers_combinations)
                facts.extend(f)

                questions_select, answers_select, f = select_value(np.random.choice(predicates),
                                                                   np.random.choice(predicates), False, False)

                questions.extend(questions_select)
                answers.extend(answers_select)
                facts.extend(f)

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_join, answers_join, f = execute_join(p_1, p_2, attributes, False, False)

                questions.extend(questions_join)
                answers.extend(answers_join)
                facts.extend(f)

                questions_closure, answers_closure, f = transitive_closure(np.random.choice(closures),
                                                                           np.random.choice(predicates), False, False)

                questions.extend(questions_closure)
                answers.extend(answers_closure)
                facts.extend(f)

                questions_preferences, answers_preferences, f = preferences(np.random.choice(predicates), predicates,
                                                                            False, False)

                questions.extend(questions_preferences)
                answers.extend(answers_preferences)
                facts.extend(f)

                questions_negative, answers_negative, f = select_by_negative_condition(np.random.choice(predicates),
                                                                                       np.random.choice(predicates),
                                                                                       predicates, False, False)

                questions.extend(questions_negative)
                answers.extend(answers_negative)
                facts.extend(f)

                questions_numeric_condition, answers_numeric_condition, f = select_by_numeric_condition(
                    np.random.choice(predicates), False, False)

                questions.extend(questions_numeric_condition)
                answers.extend(answers_numeric_condition)
                facts.extend(f)

                if print_proportions:
                    len_questions = len(questions)

                    print("ass", n_questions_assignment, n_questions_assignment * size,
                          n_questions_assignment * size / len_questions * 100)
                    print("prev", n_questions_prevent, n_questions_prevent * size,
                          n_questions_prevent * size / len_questions * 100)
                    print("comb", len(questions_combinations), len(questions_combinations) * size,
                          len(questions_combinations) * size / len_questions * 100)
                    print("join", len(questions_join), len(questions_join) * size,
                          len(questions_join) * size / len_questions * 100)
                    print("clos", len(questions_closure), len(questions_closure) * size,
                          len(questions_closure) * size / len_questions * 100)
                    print("pref", len(questions_preferences), len(questions_preferences) * size,
                          len(questions_preferences) * size / len_questions * 100)
                    print("filt", len(questions_select), len(questions_select) * size,
                          len(questions_select) * size / len_questions * 100)
                    print("neg filt", len(questions_negative), len(questions_negative) * size,
                          len(questions_negative) * size / len_questions * 100)
                    print("num filt", len(questions_numeric_condition), len(questions_numeric_condition) * size,
                          len(questions_numeric_condition) * size / len_questions * 100)
                    break

            case "core-invariance":

                results_path = "Core-Invariance/"

                exhaustive_folder = results_path + "exhaustive/"
                if not os.path.exists(exhaustive_folder):
                    os.makedirs(exhaustive_folder)

                prompt_invariance = True

                n_questions_assignment = n_questions_prevent = n_questions_combination = 0
                n_questions_join = n_questions_closure = n_questions_preferences = n_questions_select = 0
                n_questions_negative = n_questions_numeric = 0

                for _ in range(80):  # assignment
                    question_assignments, answer_assignments, f = label_assignment(predicates,
                                                                                   np.random.choice(predicates),
                                                                                   prompt_invariance, False)
                    questions.extend(question_assignments)
                    answers.extend(answer_assignments)
                    facts.extend(f)
                    n_questions_assignment += len(question_assignments)

                for _ in range(40):  # constraint
                    question_prevents, answer_prevents, f = prevent_value(predicates, np.random.choice(predicates),
                                                                          prompt_invariance, False)
                    questions.extend(question_prevents)
                    answers.extend(answer_prevents)
                    facts.extend(f)
                    n_questions_prevent += len(question_prevents)

                for _ in range(35):  # combination
                    p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                    questions_combinations, answers_combinations, f = generate_combinations(p_1, p_2, prompt_invariance,
                                                                                            False)
                    questions.extend(questions_combinations)
                    answers.extend(answers_combinations)
                    facts.extend(f)
                    n_questions_combination += len(questions_combinations)

                for _ in range(20):  # join
                    p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                    questions_join, answers_join, f = execute_join(p_1, p_2, attributes, prompt_invariance, False)
                    questions.extend(questions_join)
                    answers.extend(answers_join)
                    facts.extend(f)
                    n_questions_join += len(questions_join)

                for _ in range(40):  # closure
                    questions_closure, answers_closure, f = transitive_closure(np.random.choice(closures),
                                                                               np.random.choice(predicates),
                                                                               prompt_invariance, False)
                    questions.extend(questions_closure)
                    answers.extend(answers_closure)
                    facts.extend(f)
                    n_questions_closure += len(questions_closure)

                for _ in range(5):  # preference
                    questions_preferences, answers_preferences, f = preferences(np.random.choice(predicates),
                                                                                predicates, prompt_invariance, False)
                    questions.extend(questions_preferences)
                    answers.extend(answers_preferences)
                    facts.extend(f)
                    n_questions_preferences += len(questions_preferences)

                for _ in range(60):  # filtering
                    questions_select, answers_select, f = select_value(np.random.choice(predicates),
                                                                       np.random.choice(predicates), prompt_invariance,
                                                                       False)
                    questions.extend(questions_select)
                    answers.extend(answers_select)
                    facts.extend(f)
                    n_questions_select += len(questions_select)

                for _ in range(20):  # negative filtering
                    questions_negative, answers_negative, f = select_by_negative_condition(np.random.choice(predicates),
                                                                                           np.random.choice(predicates),
                                                                                           predicates,
                                                                                           prompt_invariance, False)
                    questions.extend(questions_negative)
                    answers.extend(answers_negative)
                    facts.extend(f)
                    n_questions_negative += len(questions_negative)

                for _ in range(20):  # numeric filtering
                    questions_numeric_condition, answers_numeric_condition, f = select_by_numeric_condition(
                        np.random.choice(predicates), prompt_invariance, False)
                    questions.extend(questions_numeric_condition)
                    answers.extend(answers_numeric_condition)
                    facts.extend(f)
                    n_questions_numeric += len(questions_numeric_condition)

                if print_proportions:
                    tot_questions = n_questions_assignment + n_questions_prevent + n_questions_combination + n_questions_join + n_questions_closure + n_questions_preferences + n_questions_select + n_questions_negative + n_questions_numeric
                    tot_questions_size = tot_questions * size
                    print("size = ", size)
                    print("tot_questions = ", tot_questions, "tot_questions_size = ", tot_questions_size)
                    print("ass", n_questions_assignment, n_questions_assignment * size,
                          n_questions_assignment * size / tot_questions_size * 100)
                    print("prev", n_questions_prevent, n_questions_prevent * size,
                          n_questions_prevent * size / tot_questions_size * 100)
                    print("comb", n_questions_combination, n_questions_combination * size,
                          n_questions_combination * size / tot_questions_size * 100)
                    print("join", n_questions_join, n_questions_join * size,
                          n_questions_join * size / tot_questions_size * 100)
                    print("clos", n_questions_closure, n_questions_closure * size,
                          n_questions_closure * size / tot_questions_size * 100)
                    print("pref", n_questions_preferences, n_questions_preferences * size,
                          n_questions_preferences * size / tot_questions_size * 100)
                    print("filt", n_questions_select, n_questions_select * size,
                          n_questions_select * size / tot_questions_size * 100)
                    print("neg filt", n_questions_negative, n_questions_negative * size,
                          n_questions_negative * size / tot_questions_size * 100)
                    print("num filt", n_questions_numeric, n_questions_numeric * size,
                          n_questions_numeric * size / tot_questions_size * 100)
                    sys.exit(1)

            case "complex":
                results_path = "BaseComplex/"

                exhaustive_folder = results_path + "exhaustive/"
                if not os.path.exists(exhaustive_folder):
                    os.makedirs(exhaustive_folder)

                n_questions_jf = n_questions_gc = n_questions_cnef = 0

                for _ in range(20):  # join filtering
                    p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                    questions_jf, answers_jf, f = join_filtering(p_1, p_2, attributes, predicates)

                    questions.extend(questions_jf)
                    answers.extend(answers_jf)
                    facts.extend(f)

                    n_questions_jf += len(questions_jf)

                for _ in range(135):  # guessing constraint
                    questions_gc, answers_gc, f = guessing_constraint(labels, np.random.choice(predicates))

                    questions.extend(questions_gc)
                    answers.extend(answers_gc)
                    facts.extend(f)

                    n_questions_gc += len(questions_gc)

                for _ in range(117):  # combination negative filtering
                    questions_cnef, answers_cnef, f = combination_negative_filtering(labels,
                                                                                     np.random.choice(predicates),
                                                                                     np.random.choice(predicates),
                                                                                     np.random.choice(predicates))

                    questions.extend(questions_cnef)
                    answers.extend(answers_cnef)
                    facts.extend(f)

                    n_questions_cnef += len(questions_cnef)

                if print_proportions:
                    len_questions = len(questions)

                    print("jf", n_questions_jf, n_questions_jf * size, n_questions_jf / len_questions * 100)
                    print("gc", n_questions_gc, n_questions_gc * size, n_questions_gc / len_questions * 100)
                    print("cnef", n_questions_cnef, n_questions_cnef * size,
                          n_questions_cnef / len_questions * 100)
                    sum = n_questions_jf + n_questions_gc + n_questions_cnef
                    print("tot = ", sum, " ", sum * size)
                    sys.exit(1)

    random.seed(42)
    temp = list(zip(questions, answers))

    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    questions, answers = list(res1), list(res2)

    return questions, answers


def main():
    turn = "core"  # "core-invariance" # "complex"

    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    match turn:
        case "core":

            train_file_name = "train_core.csv"
            val_file_name = "val_core.csv"

            tot_size = 100000

        case "core-invariance":

            train_file_name = "train_invariance.csv"
            val_file_name = "val_invariance.csv"

            tot_size = 100

        case "complex":
            train_file_name = "train_basecomplex.csv"
            val_file_name = "val_basecomplex.csv"

            tot_size = 10000

        case _:
            print("Selected Turn Not Available")
            sys.exit(1)

    size = int(0.8 * tot_size)
    val_size = int(0.2 * tot_size)

    questions, answers = generate_subproblems(turn, size, size, validation=False)
    val_questions, val_answers = generate_subproblems(turn, val_size, size, validation=True)

    print("len questions = ", len(questions))
    print("len answers = ", len(answers))

    d = {"question": questions, "answer": answers}
    val_d = {"question": val_questions, "answer": val_answers}

    train_df = pd.DataFrame(d)
    val_df = pd.DataFrame(val_d)

    train_df_fn = os.path.join(data_folder, train_file_name)
    val_df_fn = os.path.join(data_folder, val_file_name)

    train_df.to_csv(train_df_fn, index=False)
    val_df.to_csv(val_df_fn, index=False)


if __name__ == '__main__':
    main()
