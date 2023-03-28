import re
import xml.etree.ElementTree as Et


def load_civil_codes(file_path, path_data_alignment=None):
    article_elements = {}
    article_id = None
    civil_name = ""
    chapter_name = ""
    section_name = ""
    subsection_name = ""
    division_name = ""
    part_name = ""
    annotated_line = ""

    # load data alignment in english language
    if path_data_alignment is not None:
        with open(path_data_alignment, "rt") as file_align:
            data_alignment = [_l.strip() for _l in file_align.readlines()]

    # load data
    with open(file_path, "rt") as file_civil:
        for i, data_l in enumerate(file_civil.readlines()):
            data_l = data_l.strip()
            _l = data_alignment[i]

            if _l.startswith("Civil Code "):
                civil_name = data_l
                part_name, chapter_name, section_name, subsection_name, division_name, annotated_line = \
                    "", "", "", "", "", ""
            elif _l.startswith('Part '):
                part_name = data_l
                chapter_name, section_name, subsection_name, division_name, annotated_line = "", "", "", "", ""
            elif _l.startswith('Chapter '):
                chapter_name = data_l
                section_name, subsection_name, division_name, annotated_line = "", "", "", ""
            elif _l.startswith('Section '):
                section_name = data_l
                subsection_name, division_name, annotated_line = "", "", ""
            elif _l.startswith('Subsection '):
                subsection_name = data_l
                division_name, annotated_line = "", ""
            elif _l.startswith('Division '):
                division_name = data_l
                annotated_line = ""
            elif re.fullmatch(r'\([^)]*\)', _l.strip()) is not None:
                annotated_line = data_l
                # print("[W] Skip line {}".format(_l))
            elif "deleted" in _l.lower():
                pass
            elif _l.startswith("Article"):
                article_id = re.search(r"Article ([^ ]*) ", _l).group(1)

                # get article content with out id part
                article_info = data_l.split('\u3000')
                if len(article_info) > 1 and len(article_info[1].strip()) > 0:
                    article_content = article_info[1].strip()
                else:
                    article_content = data_l

                # save article
                if article_id not in article_elements:
                    article_elements[article_id] = {
                        "civil_name": civil_name,
                        "chapter_name": chapter_name,
                        "section_name": section_name,
                        "subsection_name": subsection_name,
                        "division_name": division_name,
                        "part_name": part_name,
                        "annotated_line": annotated_line,
                        "content": article_content,
                    }
                # print(article_id)
            else:
                if article_id is not None:
                    if article_id not in article_elements:
                        article_elements[article_id] = {
                            "civil_name": civil_name,
                            "chapter_name": chapter_name,
                            "section_name": section_name,
                            "subsection_name": subsection_name,
                            "division_name": division_name,
                            "part_name": part_name,
                            "annotated_line": annotated_line,
                            "content": article_content,
                        }
                    article_elements[article_id]["content"] = article_elements[article_id]["content"] + " \n" + data_l
                else:
                    print("[W] error id = {} with text = {}".format(
                        article_id, _l))

    return article_elements


def remove_article_head(s):
    l = 0
    article_search = re.search('(?<=^Article )([^ \n]+)', s)
    if article_search:
        l += len(article_search.group(0))
        l += len('Article ')
    return s[l:].lstrip()


def _parse_article_text(article_text):
    article_element = {}
    article_id = None
    for _l in article_text.split("\n"):
        if _l.startswith("Article"):
            article_id = _l[len("Article "):]
        else:
            if article_id is not None:
                if article_id not in article_element:
                    article_element[article_id] = ""
                article_element[article_id] = article_element[article_id] + " \n " + _l
            else:
                print("[W] error id = {} with text = {}".format(article_id, _l))

    return article_element


def _parse_article_fix(article_text):
    return re.findall('(?<=^Article )([^ \n]+)', article_text, re.MULTILINE)


def load_samples(filexml, file_alignment=None):
    try:
        if file_alignment is not None:
            tree_alignment = Et.parse(file_alignment)
            root_alignment = tree_alignment.getroot()
        tree = Et.parse(filexml)
        root = tree.getroot()
        samples = []
        for i in range(0, len(root)):
            sample = {'result': []}
            for j, e in enumerate(root[i]):
                if e.tag == "t1":
                    sample['result'] = _parse_article_fix(e.text.strip())
                elif e.tag == "t2":
                    question = e.text.strip()
                    sample['content'] = question if len(question) > 0 else None
            sample.update(
                {'index': root[i].attrib['id'], 'label': root[i].attrib.get('label', "N")})

            # filter the noise samples
            if sample['content'] is not None:
                samples.append(sample)
            else:
                print("[Important warning] samples {} is ignored".format(sample))

        return samples
    except Exception as e:
        print(e)
        print("[Err] parse tree error {}".format(filexml))


def check_is_usecase(q: str):
    q = q.replace(',', ' ').replace('.', ' ').replace(')',
                                                      ' ').replace(')', ' ').replace("'", ' ')
    for i in range(1, len(q) - 1):
        if q[i-1] == ' ' and q[i+1] == ' ' and q[i].isupper() and q[i] not in ['A', 'I']:
            return True
    return False
