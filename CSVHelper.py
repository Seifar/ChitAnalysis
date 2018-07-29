import csv

TARGET = "target"
SCORE = "score"
X = "x"
TEN = "10"

class Writer:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "w")
        self.writer = csv.DictWriter(self.file, fieldnames=[TARGET, SCORE, X, TEN])
        self.writer.writeheader()

    #writes one line to the file
    def write (self, dict):
        self.writer.writerow(dict)

    #writes a List of dates to the file
    def writeAll (self, dictList):
        for entry in dictList:
            self.write(entry)