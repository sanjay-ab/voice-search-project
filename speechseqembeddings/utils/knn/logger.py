import datetime
import os
import subprocess
import sys

from pathlib2 import Path


class Logger():
    def __init__(self, logDirPath=None, verbose=False,exp_name=None):
        self.instantiationTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.verbose = verbose
        
        if logDirPath:
            self.initKNNExperiment(logDirPath,exp_name)

    def initKNNExperiment(self, logDirPath,exp_name):
        self.logDirPath = logDirPath

        # Create log folder. This is where all files
        # are going to be created and stored.
        self.outputFolder=self.logDirPath
#        self.outputFolder = os.path.join(
#            self.logDirPath,
#            exp_name
#        )
        if Path(os.path.join(self.outputFolder, '.done_pairs')).is_file():
            print(self.outputFolder,"is already done")
        if not os.path.isdir(self.outputFolder):
            try:
                os.mkdir(self.outputFolder)
            except:
                pass
        print('Running experiments in %s' % self.outputFolder)

        self.logFilePath = os.path.join(self.outputFolder, 'logs')
        self.pairFilePath = os.path.join(self.outputFolder, 'pairs')
        self.pairFileFolder = os.path.join(self.outputFolder, 'pairs_folder')
        if not os.path.isdir(self.pairFileFolder):
            try:
                os.mkdir(self.pairFileFolder)
            except:
                pass
        self.filledUttPairsFilePath = os.path.join(self.outputFolder, 'filledUttPairs')
        self.clearedUttPairsFilePath = os.path.join(self.outputFolder, 'clearedUttPairs')
        self.filenameMappingFilePath = os.path.join(self.outputFolder, 'filenameMapping')

    def log(self, message, pbar=None):
        if hasattr(self, 'logFilePath'):
            with open(self.logFilePath, 'a') as f:
                f.write('{}\n'.format(message))

        if self.verbose:
            if pbar:
                pbar.write(message)
            else:
                print(message)

    def updatePbarDescription(self, description, pbar):
        if pbar:
            pbar.set_description(description)
            pbar.refresh()

    # METHODS WHICH WRITE TO FILES

    def logDoneFile(self):
        Path(os.path.join(self.outputFolder, '.done_pairs')).touch()

    def logPairs(self, boxes, filename, pbar=None):
        def parse_box(box):
            assert(len(box)==9)
            fid1,fid2, f1, f2, a1, b1, a2, b2, d = box
            #assert(float(d)>=0)
            return [fid1,f1, float(a1)/100, float(b1)/100,fid2,f2, float(a2)/100, float(b2)/100, float(d)]
            #return [fid1,fid2, f1, f2, int(a1), int(b1), int(a2), int(b2), float(d)]

        self.updatePbarDescription('Writing pairs', pbar)
        across_count=0
        output_pair_file=os.path.join(self.pairFileFolder,filename)
        with open(output_pair_file, 'w') as f:
            for box in boxes:
                pair=parse_box(box)
                f.write('{} {} {} {} {} {} {} {} {}\n'.format(*pair))

    def concatPairFiles(self):
        with open(self.pairFilePath,'a') as globalpairfile:
            for f in os.listdir(self.pairFileFolder):
                with open(os.path.join(self.pairFileFolder,f),'r') as pairfile:
                    current_pairs=pairfile.read()
                globalpairfile.write(current_pairs)


    def logFilledUttPairs(self, uttPairOccurrences):
        with open(self.filledUttPairsFilePath, 'w') as f:
            for key in uttPairOccurrences:
                f.write('{}\n'.format(str(uttPairOccurrences[key])))

    def logClearedUttPairs(self, clearedUttPairs):
        with open(self.clearedUttPairsFilePath, 'w') as f:
            for n in clearedUttPairs:
                f.write('{}\n'.format(str(n)))

    def logFilenameMapping(self, d):
        with open(self.filenameMappingFilePath, 'w') as f:
            for filename in d:
                f.write('{} {}\n'.format(d[filename], filename))
