import sqlite3

try:
    sqliteConnection = sqlite3.connect('datenbank.db')
    cursor = sqliteConnection.cursor()
    sqlite_insert_query = """INSERT INTO gen_alg_tabelle ('reprod', 'mut_prob', 'trial_num', 'gen_num', 'max_score', 'av_score', 'var_scores', 'hamm_dist') VALUES ('XXX',0.01,420,69,200,100,42.42,23.323)"""
    count = cursor.execute(sqlite_insert_query)
    sqliteConnection.commit()
    cursor.close()

except sqlite3.Error as error:
    placeholder_var = 1
finally:
    if (sqliteConnection):
        sqliteConnection.close()
