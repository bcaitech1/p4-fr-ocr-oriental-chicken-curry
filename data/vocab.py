START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]

def load_vocab(tokens_paths):
    """ token을 id화 혹은 id로 토큰을 작을 수 있는 딕셔너리 반환

    Args:
        tokens_paths(str) : token정보가 들어있는 txt파일의 경로

    Returns:
        dict : key - token, value - id인 딕셔너리
        dict : key - id, value - token인 딕셔너리
    """
    tokens = []
    tokens.extend(SPECIAL_TOKENS)
    for tokens_file in tokens_paths:
        with open(tokens_file, "r") as fd:
            reader = fd.read()
            for token in reader.split("\n"):
                if token not in tokens:
                    tokens.append(token)
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_token = {i: tok for i, tok in enumerate(tokens)}
    return token_to_id, id_to_token

def load_group_vocab(tokens_paths):
    """ token을 id화 혹은 id로 토큰을 작을 수 있는 딕셔너리 반환(그룹별-그룹1)

    Args:
        tokens_paths(str) : token정보가 들어있는 txt파일의 경로

    Returns:
        dict : key - token, value - id인 딕셔너리
        dict : key - id, value - token인 딕셔너리
    """
    tokens = []
    tokens.extend(SPECIAL_TOKENS)
    for tokens_file in tokens_paths:
        with open(tokens_file, "r") as fd:
            reader = fd.read()
            for token in reader.split("\n"):
                if token not in tokens:
                    tokens.append(token)
    
    token_to_id = {}
    id_to_token = {}

    # 토큰 번호 지정
    for i, tok in enumerate(tokens):
        if tok == "<SOS>":
            token_to_id[tok] = 0
        elif tok == "<EOS>":
            token_to_id[tok] = 1
        elif tok == "<PAD>":
            token_to_id[tok] = 2 
        elif tok.startswith('\\'):
            token_to_id[tok] = 3
        elif tok == "{":
            token_to_id[tok] = 4
        elif tok == "}":
            token_to_id[tok] = 5
        elif '0' <= tok <= '9':
            token_to_id[tok] = 6
        elif 'a' <= tok <= 'z' or 'A' <= tok <= 'Z':
            token_to_id[tok] = 7
        else:# + = - 
            token_to_id[tok] = 8

    #반환 정보 지정
    id_to_token[0] = "<SOS>"
    id_to_token[1] = "<EOS>"
    id_to_token[2] = "<PAD>"
    id_to_token[3] = "group0"
    id_to_token[4] = "group1"
    id_to_token[5] = "group2"
    id_to_token[6] = "group3"
    id_to_token[7] = "group4"
    id_to_token[8] = "group5"
    
    return token_to_id, id_to_token