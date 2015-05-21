__author__ = 'Johannes'

from os.path import join
import numpy as np

FILE_MOVIES = r"movies.dat"
FILE_USERS = r"users.dat"
FILE_RATINGS = r"ratings.dat"

BLANK = "*BLANK*"


class Movies(object):

    genreList = [BLANK, "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    genreMap = {"Action": 1,
                "Adventure": 2,
                "Animation": 3,
                "Children's": 4,
                "Comedy": 5,
                "Crime": 6,
                "Documentary": 7,
                "Drama": 8,
                "Fantasy": 9,
                "Film-Noir": 10,
                "Horror": 11,
                "Musical": 12,
                "Mystery": 13,
                "Romance": 14,
                "Sci-Fi": 15,
                "Thriller": 16,
                "War": 17,
                "Western": 18}

    def __init__(self):
        self.movieList = [(0, BLANK, [])]
        self.indexMap = {0: 0}

    def read(self, moviePath):
        for line in open(moviePath, "r"):
            movieStr = line.rstrip().split("::")
            movieObj = (int(movieStr[0]), movieStr[1], movieStr[2].split("|"))
            # in the dataset, movieIDs are not consistently numbered. There are some movieIDs missing in the dataset. That's why we create a quickndirty indexing table to look up the correct ids.
            self.indexMap[int(movieStr[0])] = len(self.movieList)
            self.movieList.append(movieObj)

    def getInternalIndex(self, idx):
        return self.indexMap[idx]

    def getMovieByInternalIdx(self, iidx):
        return self.movieList[iidx]

    def getMovieByExternalIdx(self, eidx):
        return self.movieList[self.indexMap[eidx]]

    def size(self):
        return len(self.movieList)


class Users(object):

    agePlainList = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    ageList = [BLANK]*57
    ageList[1] = "Under 18"
    ageList[18] = "18-24"
    ageList[25] = "25-34"
    ageList[35] = "35-44"
    ageList[45] = "45-49"
    ageList[50] = "50-55"
    ageList[56] = "56+"

    ageMap = {"Under 18": 1,
              "18-24": 18,
              "25-34": 25,
              "35-44": 35,
              "45-49": 45,
              "50-55": 50,
              "56+": 56}

    occupationList = ["other", "academic/educator", "artist", "clerical/admin", "college/grad student", "customer service", "doctor/health care", "executive/managerial", "farmer", "homemaker", "K-12 student", "lawyer", "programmer", "retired", "sales/marketing", "scientist", "self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"]
    occupationMap = {"other": 0,
                     "academic/educator": 1,
                     "artist": 2,
                     "clerical/admin": 3,
                     "college/grad student": 4,
                     "customer service": 5,
                     "doctor/health care": 6,
                     "executive/managerial": 7,
                     "farmer": 8,
                     "homemaker": 9,
                     "K-12 student": 10,
                     "lawyer": 11,
                     "programmer": 12,
                     "retired": 13,
                     "sales/marketing": 14,
                     "scientist": 15,
                     "self-employed": 16,
                     "technician/engineer": 17,
                     "tradesman/craftsman": 18,
                     "unemployed": 19,
                     "writer": 20}

    def __init__(self):
        self.users = [(0, BLANK, BLANK, BLANK, BLANK)]

    def read(self, userPath):
        for line in open(userPath, "r"):
            userStr = line.rstrip().split("::")
            self.users.append((int(userStr[0]), userStr[1], self.ageList[int(userStr[2])], self.occupationList[int(userStr[3])], userStr[4]))

    def getUser(self, idx):
        return self.users[idx]

    def size(self):
        return len(self.users)


class Ratings(object):

    def __init__(self):
        self.ratings = []

    def read(self, ratingPath):
        for line in open(ratingPath, "r"):
            ratingStr = line.rstrip().split("::")
            self.ratings.append((int(ratingStr[0]), int(ratingStr[1]), int(ratingStr[2]), ratingStr[3]))

    def getRating(self, idx):
        return self.ratings[idx]

    def size(self):
        return len(self.ratings)


class MovielensDataset(object):

    def __init__(self, rootPath):
        self.movies = Movies()
        self.movies.read(join(rootPath, FILE_MOVIES))
        self.users = Users()
        self.users.read(join(rootPath, FILE_USERS))
        self.ratings = Ratings()
        self.ratings.read(join(rootPath, FILE_RATINGS))

    def getRatingsOfUser(self, userIdx):
        userratings = []
        for r in self.ratings.ratings:
            if r[0] == userIdx:
                userratings.append((self.movies.getMovieByExternalIdx(r[1]), r[2]))
        return userratings

    def getCategoriesOfUser(self, userIdx):
        userCats = {}
        for r in self.ratings.ratings:
            if r[0] == userIdx:
                for m in self.movies.getMovieByExternalIdx(r[1])[2]:
                    if m in userCats:
                        userCats[m] += 1
                    else:
                        userCats[m] = 1
        return userCats

    def getCategoriesOfUsers(self, userList):
        userCats = {}
        totCount = 0
        for userId in userList:
            for r in self.ratings.ratings:
                if r[0] == userId:
                    for m in self.movies.getMovieByExternalIdx(r[1])[2]:
                        totCount += 1
                        if m in userCats:
                            userCats[m] += 1
                        else:
                            userCats[m] = 1
        for k in userCats:
            userCats[k] = float(userCats[k]) / float(totCount)
        return userCats

    def getUserItemMatrix(self):
        uimat = np.zeros((self.users.size(), self.movies.size()))
        print self.movies.size()
        print self.users.size()
        print self.ratings.size()

        for r in self.ratings.ratings:
            uimat[r[0], self.movies.getInternalIndex(r[1])] = r[2]/5.0#2.0 * (r[2] / 5.0 - 0.5)
        return uimat

    def getUsersAttributesMatrix(self):
        uamat = np.zeros((self.users.size(), 30))
        for i in xrange(self.users.size()):
            if i == 0:
                binaryUser = np.zeros((1, 30))
            else:
                userObj = self.users.getUser(i)
                binarySex = np.zeros((1,2))
                if userObj[1] == 'M':
                    binarySex[0,0] = 1
                else:
                    binarySex[0,1] = 1
                binaryAge = np.zeros((1, len(self.users.agePlainList)))
                binaryAge[0, self.users.agePlainList.index(userObj[2])] = 1
                binaryOcc = np.zeros((1, len(self.users.occupationList)))
                binaryOcc[0, self.users.occupationList.index(userObj[3])] = 1
                binaryUser = np.hstack((binarySex, binaryAge, binaryOcc))
            uamat[i] = binaryUser
        a = ['M', 'F'] + self.users.agePlainList + self.users.occupationList
        for i,e in enumerate(a):
            print "%d \t %s" % (i,e)
        return uamat
