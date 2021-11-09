import re
from datetime import datetime


def anime_title_f(bs):
    # Extract text within tag denoted by a certain class
    tag_of_inter = bs.find('h1', {'class': 'title-name h1_bold_none'})
    return tag_of_inter.text


def anime_type_f(bs):
    # Extract type of anime
    return str(bs.find(text="Type:").findNext('a').contents[0])


def anime_num_episode_f(bs):
    d = bs.find("span", text=re.compile("Episodes")).next_sibling
    try:
        r = int(d.strip())
    except:
        return None
    return r


def get_date(date):
    if not any(char.isdigit() for char in date):
        return None
    d = len(date.replace(",", "").split(" "))
    if d == 3:
        release = datetime.strptime(date.replace(",", ""), '%b %d %Y')
    elif d == 2:
        release = datetime.strptime(date.replace(",", ""), '%b %Y')
    elif d == 1:
        release = datetime.strptime(date.replace(",", ""), '%Y')
    return date


def release_date_f(bs):
    dates = bs.find("span", text=re.compile("Aired")).next_sibling.strip().split(" to ")
    if len(dates) == 1:
        release = get_date(dates[0])
        end = None
    elif len(dates) == 2:
        release = get_date(dates[0])
        end = get_date(dates[1])
    else:
        return None, None
    return release, end


def anime_num_members_f(bs):
    tag_of_inter = bs.find_all('span', {'class': 'numbers members'})
    p = re.findall(r"(?<=<strong>).+(?=</strong>)", str(tag_of_inter))[0]
    return int(p.replace(",", ""))


def anime_score_f(bs):
    score = bs.find('div', class_='fl-l score').find('div').text
    return None if score == "N/A" else float(score)


def anime_users_f(bs):
    tag_of_inter = bs.find('div', {'class': 'fl-l score'})
    users = tag_of_inter.get('data-user').split(" ")[0]
    return None if users == "-" else int(users.replace(",", ""))


def anime_rank_f(bs):
    tag_of_inter = bs.find_all('span', {'class': 'numbers ranked'})
    rank = int(re.findall(r"(?<=<strong>#).+(?=</strong>)", str(tag_of_inter))[0])
    return rank


def anime_popularity_f(bs):
    tag_of_inter = bs.find_all('span', {'class': 'numbers popularity'})
    pop = int(re.findall(r"(?<=<strong>#).+(?=</strong>)", str(tag_of_inter))[0])
    return pop


def anime_description_f(bs):
    synopsis = bs.find('p', itemprop='description').text.replace("\n", "")
    return synopsis


def anime_related_f(bs):
    related_anime = set()
    tag = bs.find('table', {'class': 'anime_detail_related_anime'})
    if tag is None:
        return None
    urls = tag.find_all('a')
    if urls is None:
        return None
    for e in urls:
        related_anime.add(e.get("href"))
    return list(related_anime)


def anime_characters_f(bs):
    tags = bs.find_all('h3', {'class': 'h3_characters_voice_actors'})
    characters = [el.text for el in tags]
    return characters


def people_divs(bs, caller):
    divs = bs.find_all('div', {'class': 'detail-characters-list clearfix'})
    if divs is not None:
        l = len(divs)
        if l == 1 and caller == "voices":
            return divs[0]
        if l == 2 and caller == "staff":
            return divs[1]
        return None


def anime_voices_f(bs):
    div = people_divs(bs, "voices")
    if div is None:
        return None
    voices = [el.text for el in div.find_all('a', href=re.compile("people")) if el.text != "\n\n"]
    return voices


def anime_staff_f(bs):
    div = people_divs(bs, "staff")
    if div is None:
        return None
    staff = [el.text for el in div.find_all('a', href=re.compile("people")) if el.text != "\n\n"]
    role = [el.text for el in div.find_all('small')]
    staff = [[s, r] for s, r in zip(staff, role)]

    return staff


'''Related Anime (to save as animeRelated): Extract all the related animes, but only keep unique values 
and those that have a hyperlink associated to them. List of strings.

Characters (to save as animeCharacters): List of strings.
Voices (to save as animeVoices): List of strings
Staff (to save as animeStaff): Include the staff name and their responsibility/task in a list of lists.'''
