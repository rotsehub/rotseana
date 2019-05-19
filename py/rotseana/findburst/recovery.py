'''
Created on Jul 22, 2017

@author: arnon
'''
import os
import pickle


class Recovery(object):
    def __init__(self, name, assoc_path, location=None):
        '''
        Args:
            name (str): unique name to id the object
            assoc_path (str) : the associated match file
            location (path): place to store recovery objects, use match_file location if None.
        '''
        self.assoc_path = assoc_path
        self.name = name
        self.obj_file = self.get_obj_file(location)

    def get_obj_file(self, location=None):
        result = self.assoc_path+'.%s' % self.name
        if location:
            name = os.path.basename(result)
            result = os.path.join(location, name)
        return result

    def load(self,):
        obj = None
        if self.obj_file:
            if os.path.isfile(self.obj_file):
                obj_file_m_time = os.path.getmtime(self.obj_file)
                assoc_path_m_time = 0
                if os.path.isfile(self.assoc_path) or os.path.isdir(self.assoc_path):
                    assoc_path_m_time = os.path.getmtime(self.assoc_path)
                if assoc_path_m_time > 0 and obj_file_m_time >= assoc_path_m_time:
                    # not a new file, read goodobj from file
                    print("Recovering %s from %s" % (self.name, self.obj_file))
                    with open(self.obj_file, 'rb') as f:
                        obj = pickle.load(f)
        return obj

    def store(self, obj):
        print("Storing %s into %s" % (self.name, self.obj_file))
        with open(self.obj_file, 'wb') as f:
            pickle.dump(obj, f)
