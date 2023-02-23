import hashlib
import numpy as np
import csv
import pandas as pd


def create_attack_table():
    
    with open("rockyou.txt") as file:
        breached_passwords = file.readlines()

    breached_passwords = [password.rstrip() for password in breached_passwords]
    
    csv_file = open("attack_table.csv", "w", newline='')
    csv_writer = csv.writer(csv_file, delimiter=",")

    for password in breached_passwords:
        csv_writer.writerow([password, hashlib.sha512(str.encode(password)).hexdigest()])

    csv_file.close()


def find_passwords():
    attack_table = pd.read_csv("attack_table.csv").to_numpy()
    lookup_table = {h:p for p,h in attack_table}
    
    digital_corp_data = pd.read_csv("digitalcorp.txt").to_numpy()

    for user in digital_corp_data:
        if user[1] in lookup_table:
            print(f"{user[0]}'s password is {lookup_table[user[1]]}")


def find_salty_passwords():
    attack_table = pd.read_csv("attack_table.csv").to_numpy()
    vanilla_lookup_table = {h:p for p,h in attack_table}
    
    salty_digital_corp_data = pd.read_csv("salty-digitalcorp.txt").to_numpy()
    
    for user in salty_digital_corp_data:
        salty_lookup_table = {}
    
        for password in vanilla_lookup_table.values():
            salty_password = user[1] + password
            salty_hash = hashlib.sha512(str.encode(salty_password)).hexdigest()
            salty_lookup_table[salty_hash] = password
    
        if user[2] in salty_lookup_table:
            print(f"{user[0]}'s password is {salty_lookup_table[user[2]]}")



if __name__ == "__main__":
    create_attack_table()
    find_passwords()
    find_salty_passwords()