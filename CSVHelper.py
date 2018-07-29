import csv

class Writer:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "w")
        self.writer = csv.writer(self.file)
        self.write(["target number", "score", "x", "10"])

    #writes one line to the file
    def write (self, data):
        print("writing: "+ str(data))
        self.writer.writerow(data)

    #writes a List of dates to the file
    def writeAll (self, data):
        for entry in data:
            self.write(entry)