import csv
import glob
import multiprocessing as mp
import os
import re
import bs4
import parse_utils as p


# task function to get tsv files
def parse_anime(starting_page, path):
    # each task works on 28 pages
    task_dim = 28
    # last task processes 19 pages instead of 28
    if starting_page == 364:
        task_dim = 19
    for i in range(starting_page, starting_page + task_dim):
        # get the path to the dir where all the folders are stored
        parent_dir = os.path.join(path, "data")
        dir_name = os.path.join(parent_dir, f"page_{i}")
        # for each folder, create a folder for its tvs files
        home = os.path.join(dir_name, f"animes_{i}")
        os.makedirs(home)
        os.chdir(home)
        index = i*50
        # for each html file in the dir create a tsv and write some extracted information there
        for file in glob.iglob(r'' + re.escape(dir_name)+ '\*.html'):
            # open and read the html file to get the info
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
                # Write a csv with the data collected
                with open(f'anime_{index}.tsv', 'w', newline='', encoding='utf8') as f_output:
                    tsv_output = csv.writer(f_output, delimiter='\t')
                    tsv_output.writerow(list(info.keys()))
                    tsv_output.writerow(list(info.values()))
                print("tsv_" + str(index) + " created")
            index += 1


# function to handle parallelization
def parallelize_parsing(path):
    # create a pool to be later populated and define a coherent number of tasks to exploit parallelization better
    pool = mp.Pool(mp.cpu_count())
    # 14 tasks - 13 * 28 + 18 : 28 pages per task ( last excluded, takes 18 )
    page = 0
    # start a task every 28 pages
    for r in range(0, 383, 28):
        pool.apply(parse_anime, args=(page + r, path))
    # close the pool
    pool.close()

