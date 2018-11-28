# coding: utf-8
import json, re
import logging
logging.basicConfig(level = logging.INFO)


def load_jsonlines(filename) :
    with open(filename) as f :
        for line in f :
            yield json.loads(line)

def return_token(scanner, token) :
    return token
def return_tagged_token(tag) :
    def h(scanner, token) :
        return (tag, token)
    return h
def unexpected_token(scanner, token) :
    raise ValueError("unexpected token: '{}'".format(token))

BEGIN_SYMBOL, END_SYMBOL = '<', '>'
scanner_rules = [
    (r"(a|ɑ|æ|ɜ|e|ʌ|ə|ɒ|ɔ|o|i|u):?|(ɪ|ʊ)", return_token),
    (r"p|b|m|f|v|θ|ð|s|z|n|tʃ?|dʒ?|ʃ|ʒ|k|g|ŋ|h|w|l|r|j", return_token),
    (r"'|ˌ|/", None),
    (r".", unexpected_token),
]
scanner = re.Scanner(scanner_rules)

def main() :
    import argparse, itertools, os

    parser = argparse.ArgumentParser(description =
        '''Parse phonemic scripts into phonemes inventory and lists of phonemes.''')
    parser.add_argument('--input', required = True,
        help='*.jsonlines containing entries with "phonemic-script" and "word" fields.')
    parser.add_argument('--output-dir', required = True,
        help='output directory used to save phoneme inventory, words and parsed transcripts.')
    args = parser.parse_args()


    entries = [entry
    for entry in load_jsonlines(args.input)
    if entry.get('phonemic-script') and entry.get('word')
    ]


    transcripts = [entry['phonemic-script'] for entry in entries]
    words = [entry['word'] for entry in entries]

    results = list()
    for t, w in zip(transcripts, words) :
        try :
            result, reminder = scanner.scan(t)
        except ValueError as e :
            logging.info("While processing for word \"{}\"".format(w))
            raise e
        assert (len(reminder) == 0)
        results.append(result)

    phonemes = list(sorted(set(itertools.chain(*(r for r in results)))))
    inventory = [BEGIN_SYMBOL, END_SYMBOL] + phonemes
    inventory_lookup = dict(zip(inventory, itertools.count()))


    os.makedirs(args.output_dir, exist_ok = True)

    with open(os.path.join(args.output_dir, 'inventory.list(string).txt'), 'w') as f :
        for x in inventory :
            print(x, file = f)
    with open(os.path.join(args.output_dir, 'words.list(string).txt'), 'w') as f :
        for word in words :
            print(word, file = f)
    with open(os.path.join(args.output_dir, 'transcripts.list(list(id)).txt'), 'w') as f :
        bs_id, es_id = inventory_lookup[BEGIN_SYMBOL], inventory_lookup[END_SYMBOL]
        for ps in results :
            p_ids = [bs_id] + [inventory_lookup[p] for p in ps] + [es_id]
            print(*p_ids, file = f)

if __name__ == '__main__' :
    main()
