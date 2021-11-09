import os
import time
import list_urls as l_u
import content_html as c_html
import anime_information as a_info

if __name__ == '__main__':
    start_time = time.time()

    print("Starting...")
    # get the txt file with all the urls
    l_u.parallelize_extraction()
    # get the html files
    c_html.get_content()
    # get the info about the animes
    a_info.parallelize_parsing(os.getcwd())

    print("--- %s seconds ---" % (time.time() - start_time))