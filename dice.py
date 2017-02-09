
# coding: utf-8

# In[1]:

from __future__ import unicode_literals
from rtmbot.core import Plugin


# In[ ]:
#RNGesus
class myDice(Plugin):
    
    def process_message(self, data):

        #dice
        if data['text'][:5].lower() == '!roll':
            def dice():
                import random

                user = data['user']
                command = data['text'][-((len(data['text'].strip())) - (data['text'].strip().find(' '))-1):]

                if command.strip().lower() == '!roll':
                    roll = random.randint(1,100)
                    self.outputs.append(["C3ZMR09SN","<@{}> rolled a {} (1-100)".format(user,roll)])
                    #print roll

                else: 
                    try:
                        if type(int(command)) == int:
                            roll = random.randint(1,int(command))
                            if roll == int(command):
                                comment = ['. RNGesus smiles down on you.','. Blessings from RNGesus.','. A wild RNGesus appears!','. Thy Lord, RNGesus, giveth.','. Receive the boon of RNGesus.'][random.randint(0,4)]
                            else:
                                comment = ''
                            msg = "<@{}> rolled a {} (1-{}){}".format(user,roll,command,comment)
                            self.outputs.append(["C3ZMR09SN",msg])
                            #print roll
                    except ValueError:
                        self.outputs.append(["C3ZMR09SN", 'Curious... :thinking_face: It appears that I do not have dice that supports *{}*. Please change *"{}"* to an integer.'.format(type(command),command)])
                        #print 'oops'
            dice()
        else:
            pass


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



