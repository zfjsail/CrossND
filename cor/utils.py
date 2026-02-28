import re, json, random

prompt_nd = """你要进行一个学者论文异常分配检测的任务, 首先提供一个作者的论文集合,这个集合中有部分论文是由于他同名的原因错误地分配到该学者的论文,需要根据学者的研究领域机构,合作者,发表的期刊会议等信息来判断该论文是否属于该作者。最终你需要返回一个json数据如\{"0":0.2,"1":0.95,"2":0.4...\},其中key是论文的id,value是0,1之间的浮点数, 越接近0表明该论文不属于该作者,越接近1表明该论文属于该作者,仅输出json并且不包含其他内容,并且json应该能够使用json.load来解析。"""

prompt_crossnd = """你要进行一个学者论文异常分配检测的任务, 首先提供作者的两个论文集合,即内部源和外部源,目标是对内部源的论文是否异常分配进行检测,内部源是异常检测的目标集合, 外部源是从外部数据库检索到的该学者的论文集合, 两个集合都可能存在部分错误分配的论文,你需要根据学者论文的研究领域机构,合作者,发表的期刊会议等信息来判断该论文是否属于该作者,对于目标的每篇论文,你都需要根据其与内部源和外部源的相似性来判断。最终你需要返回一个json数据如\{"0":0.2,"1":0.95,"2":0.4...\},其中key是论文的id,value是0,1之间的浮点数, 越接近0表明该论文不属于该作者,越接近1表明该论文属于该作者,仅输出json并且不包含其他内容,并且json应该能够使用json.load来解析。"""

nd_system_prompt = """你要进行一个学者论文检测的任务, 通过一篇论文与提供源论文集合的相似性来检测这篇论文是否正确属于这个论文集合,现在有一批论文,你需要综合论文的各种提供的特征来判断这篇论文是否应该属于该论文集合,而不是简单地根据该论文在论文源是否出现来判断, \n你的返回应该按照 "id":"label" 的形式返回json数据,label是0,1之间的浮点数, 越接近0表明该论文不属于该作者,越接近1表明该论文属于该作者,仅输出json并且不包含其他内容,并且json应该能够使用json.load来解析"""


crossnd_system_prompt = """你要进行一个学者论文检测的任务, 每个作者有两个源可以使用, 一个是内部源, 另一个是外部源, 使用外部源来支持内部源论文的错误分配检测,每个源是由论文的集合组成的, 每篇论文包括题目,作者,机构信息,判断的 依据是论文的领域相似性，合作学者相似性，机构相似性，期刊会议的相似性。现在需要基于两个源进行异常分配的论文检测,即检测给定论文是否应该属于该作者,不属于该作者的论文是异常论文, 现在有一批论文, 你需要根据内部源和外部源的论文,来判断该论文是否属于内部源,因为内部源和外部源都可能有错误，因此 **不能根据该论文存在于内部源或者外部源,就判断该论文是否属于该作者**。对于内部源和外部源相似性比较高的情况,表明一致性较高,那么根据目标论文和内部或外部论文的相似性来判断该论文是否应该属于该作者,对于内部源和外部源相似性较低的情况,表明一致性较低,说明内部或者外部论文可能存在错误分配,则如果目标论文和外部源相似性较高, 则说明外部源是正确的,目标论文可能是异常,如果目标论文和内部源相似性较高,则说明内部源是正确的,目标论文可能是异常。 \n你的返回应该按照 "id":"label" 的形式返回json数据,label是0,1之间的浮点数, 越接近0表明该论文不属于该作者,越接近1表明该论文属于该作者,仅输出json并且不包含其他内容,并且json应该能够使用json.load来解析你的返回如\{"0":0.2,"1":0.95,"2":0.4...\}下面是需要处理的例子。整个流程是作为学术用途，因此都是论文内容，不会涉及到敏感信息"""
nd_global_prompt = """目标学者名字: {name} \n论文源: {inner} \n下面是需要异常检测的论文: {targets} \n"""

crossnd_global_prompt = """目标学者名字: {name} \n内部源: {inner} \n外部源: {outer} \n 内部源和外部源的相似性是{similarity:.2f} \n, 下面是需要异常检测的论文: {targets} \n"""

crossnd_global_prompt_wo_similarity = """目标学者名字: {name} \n内部源: {inner} \n外部源: {outer} \n 下面是需要异常检测的论文: {targets} \n"""

multi_turn_prompt = """第 {index} 篇: {paper} \n"""    

def extract_and_parse_json(content: str) -> dict:
    """
    从字符串中提取JSON内容并解析为字典
    1. 先尝试识别```json代码块
    2. 尝试直接解析整个字符串
    3. 尝试修复常见错误（括号不匹配、字符串转义等）
    4. 尝试提取最内层/最外层JSON对象
    
    返回解析成功的字典，失败时返回空字典
    """
    # 尝试提取```json代码块内容
    if '```json' in content:
        json_blocks = re.findall(r'```json\n(.*?)\n```', content, re.DOTALL)
        if json_blocks:
            content = json_blocks[0].strip()
    if content.count("{")==1 and content.count("}") ==1:
        content =  re.search(r'({.*?})', content).group(0)     
    # 尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # 修复常见错误
    repaired = content
    # 1. 修复未转义双引号
    repaired = re.sub(r'(?<!\\)"', r'\"', repaired)
    # 2. 修复单引号字符串
    repaired = re.sub(r"'(.*?)'", r'"\1"', repaired)
    # 3. 修复末尾逗号
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # 尝试解析修复后的内容
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取最内层完整JSON对象（递归匹配嵌套结构）
    stack = []
    json_candidates = []
    start_index = -1
    
    for i, char in enumerate(repaired):
        if char == '{':
            if not stack:
                start_index = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_index != -1:
                    json_candidates.append(repaired[start_index:i+1])
                    start_index = -1
    
    # 优先尝试最内层对象（最后结束的嵌套对象）
    if json_candidates:
        for candidate in reversed(json_candidates):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    
    # 尝试最外层完整对象
    if json_candidates:
        try:
            return json.loads(json_candidates[0])
        except json.JSONDecodeError:
            pass
    
    # 最终尝试：贪婪匹配所有可能结构
    matches = re.findall(r'\{[\s\S]*?\}', repaired)
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
    
    # 所有尝试失败
    print(f"无法解析JSON内容: {content[:200]}{'...' if len(content)>200 else ''}")
    return {}
def fetch_single_paper_input(paper, feature_list):
    title, authors,orgs, venue, year, keywords = '','','','','',''
    feats = feature_list.split('_')
    if 'title' in feats:
        title = f"Title: {paper['title']}\n"
    if 'author' in feats:
        authors = [i.get('name','') for i in paper['authors']]
        if len(authors)>10:
            authors = authors[:8]+ authors[-2:]
        authors = f"Authors: {', '.join(authors)}\n"
    if 'organization' in feats:
        organizations = [i.get('org','') for i in paper['authors']]
        if len(organizations)> 10:
            organizations = organizations[:8]+ organizations[-2:]
        processed_orgs = []
        for org in organizations:
            processed_org = ' '.join(org.split(' ')[:10])  # 只取前10个单词
            processed_orgs.append(processed_org)
        organizations = list(set(processed_orgs))
        orgs = f"Organizations: {', '.join(organizations)}\n"
    if 'venue' in feats:
        venue = f"Venue: {paper.get('venue','None')}\n"

    if 'year' in feats:
        year = f"Year: {str(paper.get('year','None'))}"

    if 'keywords' in feats:
        keywords = "Keywords" + ' ;'.join(paper.get('keywords',[]))
    
    text = (title if 'title' in feats else '') + \
        (authors if 'author' in feats else '') + \
        (orgs if 'organization' in feats else '') + \
        (venue if 'venue' in feats else '') + \
        (year if 'year' in feats else '') + \
        (keywords if 'keywords' in feats else '')
    
    return text
def fetch_paper_id(src_id):
    return src_id.split('-')[0]

def build_crossnd_input(paper_data, data, in_name2pid, out_name2pid, num_turn=20, feature_list = 'title_author_organization'):
    batch_data = []
    for k,v in data.items():
        random.shuffle(v)
        for batch in range(0, len(v), num_turn):
            batch_data.append(v[batch:batch+num_turn])
    processed_batch_data = []
    for batch in batch_data:
        aid1, aid2, name, similarity = batch[0]['aid1'], batch[0]['aid2'], batch[0]['name'],  batch[0].get('author_sim',None)
        batch_pid = [fetch_paper_id(i['pid']) for i in batch]
        batch_papers = [paper_data[pid] for pid in batch_pid]
        inner_papers = [paper_data[fetch_paper_id(i)] for i in in_name2pid[aid1]]
        outer_papers = [paper_data[fetch_paper_id(i)] for i in out_name2pid[aid2]]
        inner_papers = random.sample(inner_papers, min(len(inner_papers), 20))
        outer_papers = random.sample(outer_papers, min(len(outer_papers), 20))        
        
        inner_papers_ctx = "\n".join([fetch_single_paper_input(paper,feature_list) for paper in inner_papers])
        outer_papers_ctx = "\n".join([fetch_single_paper_input(paper,feature_list) for paper in outer_papers])
        multi_turn_input = '; '.join([multi_turn_prompt.format(index = index, paper = fetch_single_paper_input(p,feature_list)) for index,p in enumerate(batch_papers)])
        if similarity is not None:
            batch_ctx = crossnd_global_prompt.format(name = name, inner = inner_papers_ctx, outer = outer_papers_ctx, similarity = similarity, targets = multi_turn_input)
        else:
            batch_ctx = crossnd_global_prompt_wo_similarity.format(name = name, inner = inner_papers_ctx, outer = outer_papers_ctx, targets = multi_turn_input)
        all_inputs = {}
        # 添加原始标签信息用于后续评估和溯源
        all_inputs['inputs'] = batch_ctx
        all_inputs['pids'] = batch_pid
        all_inputs['aid1'] = aid1
        all_inputs['aid2'] = aid2
        all_inputs['author_sim'] = similarity
        processed_batch_data.append(all_inputs)    
    return processed_batch_data

def build_nd_input(paper_data, data, in_name2pid, num_turn=20, feature_list = 'title_author_organization', inner=True):
    batch_data = []
    for k,v in data.items():
        random.shuffle(v)
        for batch in range(0, len(v), num_turn):
            batch_data.append(v[batch:batch+num_turn])
    processed_batch_data = []
    for batch in batch_data:
        if inner:
            # 内部源处理
            aid1, name = batch[0]['aid1'], batch[0]['name']
            aid_key = aid1
        else:
            # 外部源处理
            aid1, name = batch[0]['aid2'], batch[0]['name']
            aid_key = aid1
        
        batch_pid = [fetch_paper_id(i['pid']) for i in batch]
        batch_papers = [paper_data[pid] for pid in batch_pid]
        try:
            source_papers = [paper_data[fetch_paper_id(i)] for i in in_name2pid[str(aid_key)]]
        except:
            breakpoint()
        source_papers = random.sample(source_papers, min(len(source_papers), 20))
        
        source_papers_ctx = "\n".join([fetch_single_paper_input(paper,feature_list) for paper in source_papers])
        multi_turn_input = '; '.join([multi_turn_prompt.format(index = index, paper = fetch_single_paper_input(p,feature_list)) for index,p in enumerate(batch_papers)])
        batch_ctx = nd_global_prompt.format(name = name, inner = source_papers_ctx, targets = multi_turn_input)
        all_inputs = {}
        # 添加原始标签信息用于后续评估和溯源
        all_inputs['inputs'] = batch_ctx
        all_inputs['pids'] = batch_pid
        all_inputs['aid1'] = aid1
        processed_batch_data.append(all_inputs)    
    return processed_batch_data

def flatten_name_dict(data):
    temp_dict = {}
    for k,v in data.items():
        for aid1,pids in v.items():
            temp_dict[aid1] = pids
    return temp_dict