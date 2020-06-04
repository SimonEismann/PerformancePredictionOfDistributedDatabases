"""
Simple script that deletes all files that are not necessary for our analysis.
"""
import os
import shutil

if __name__ == "__main__":
    folder = r"C:\Users\Johannes\Desktop\combined"
    for root, dirs, files in os.walk(folder):
        for name in dirs:
            f = os.path.join(root, name)
            if f.__contains__("plots") | f.__contains__("archiv"):
                print("Removing folder", f)
                shutil.rmtree(f)
                break
            dir = os.listdir(f)
            if len(dir) == 0:
                # folder is empty and can be deleted
                print("Removing empty folder", f)
                shutil.rmtree(f)
        for name in files:
            file = os.path.join(root, name)
            if file.endswith("\load.txt"):
                # This file is kept
                pass
            else:
                print("Removing file", file)
                os.remove(file)
    print("Done removing.")
