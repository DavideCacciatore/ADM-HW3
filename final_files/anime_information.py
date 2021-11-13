import csv
import glob
import multiprocessing as mp
import os
import re
import bs4
import parse_utils as p


def parse_anime(starting_page, path):
    print(starting_page, "starting from here")
    task_dim = 28
    # last task processes less pages
    if starting_page == 364:
        print()
        task_dim = 19
    for i in range(starting_page, starting_page + task_dim):
        print(i)
        # here you put the name of the dir where you have all the folders
        parent_dir = os.path.join(path, "data")
        dir_name = os.path.join(parent_dir, f"page_{i}")
        # for each folder you create a folder for its tvs files
        home = os.path.join(dir_name, f"animes_{i}")
        os.makedirs(home)
        os.chdir(home)
        index = i*50
        for file in glob.iglob(r'' + re.escape(dir_name)+ '\*.html'):
            #print(index)
            with open(file, encoding="utf-8") as fp:
                soup = bs4.BeautifulSoup(fp, "html.parser")
                with open(file, encoding="utf-8") as fp:
                    soup = bs4.BeautifulSoup(fp, "html.parser")
                    info = dict()
                    info["animeTitle"] = p.anime_title_f(soup)
                    info["animeType"] = p.anime_type_f(soup)
                    info["animeNumEpisode"] = p.anime_num_episode_f(soup)
                    info["releaseDate"], info["endDate"] = p.release_date_f(soup)
                    info["animeNumMembers"] = p.anime_num_members_f(soup)
                    info["animeScore"] = p.anime_score_f(soup)
                    info["animeUsers"] = p.anime_users_f(soup)
                    info["animeRank"] = p.anime_rank_f(soup)
                    info["animePopularity"] = p.anime_popularity_f(soup)
                    info["animeDescription"] = p.anime_description_f(soup)
                    info["animeRelated"] = p.anime_related_f(soup)
                    info["animeCharacters"] = p.anime_characters_f(soup)
                    info["animeVoices"] = p.anime_voices_f(soup)
                    info["animeStaff"] = p.anime_staff_f(soup)
                    # for i in info:
                    #    print(i, info[i], type(info[i]))
                    # Write a csv
                    with open(f'anime_{index}.tsv', 'w', newline='', encoding='utf8') as f_output:
                        tsv_output = csv.writer(f_output, delimiter='\t')
                        tsv_output.writerow(list(info.keys()))
                        tsv_output.writerow(list(info.values()))
                    print("tsv_" + str(index) + " created")
            index += 1


def parallelize_parsing(path):
    pool = mp.Pool(mp.cpu_count())
    # 28 tasks : 4 pages per task
    page = 0
    for r in range(0, 383, 28):
        pool.apply(parse_anime, args=(page + r, path))
    pool.close()


if __name__ == '__main__':
    parallelize_parsing(os.getcwd())
