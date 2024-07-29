import mido
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st
from matplotlib.colors import colorConverter


# inherit the origin mido class
class MidiFile(mido.MidiFile):

    def __init__(self, filename):

        mido.MidiFile.__init__(self, filename)
        self.sr = 1
        self.meta = {}
        self.events = self.get_events()

    def get_events(self):
        mid = self
        # print(mid)

        # There is > 16 channel in midi.tracks. However there is only 16 channel related to "music" events.
        # We store music events of 16 channel in the list "events" with form [[ch1],[ch2]....[ch16]]
        # Lyrics and meta data used a extra channel which is not include in "events"

        events = [[] for x in range(16)]

        # Iterate all event in the midi and extract to 16 channel form
        for track in mid.tracks:
            for msg in track:
                try:
                    channel = msg.channel
                    events[channel].append(msg)
                except AttributeError:
                    try:
                        if type(msg) != type(mido.UnknownMetaMessage):
                            self.meta[msg.type] = msg.dict()
                        else:
                            pass
                    except:
                        print("error",type(msg))

        return events

    def get_roll(self, rednoteidx):
        events = self.get_events()
        # Identify events, then translate to piano roll
        # choose a sample ratio(sr) to down-sample through time axis
        sr = self.sr

        # compute total length in tick unit
        length = self.get_total_ticks()

        # allocate memory to numpy array
        roll = np.zeros((32, 120, length // sr), dtype="int8")

        # use a register array to save the state(no/off) for each key
        note_register = [int(-1) for x in range(128)]

        # use a register array to save the state(program_change) for each channel
        timbre_register = [1 for x in range(16)]

        noteidx=0

        for idx, channel in enumerate(events):

            time_counter = 0
            volume = 100
            # Volume would change by control change event (cc) cc7 & cc11
            # Volume 0-100 is mapped to 0-127

            #print("channel", idx, "start")
            for msg in channel:
                if msg.type == "control_change":   # control=64: 서스테인페달 control=67: 소프트페달
                    if msg.control == 7:
                        volume = msg.value
                        # directly assign volume
                    if msg.control == 11:
                        volume = volume * msg.value // 127
                        # change volume by percentage
                    # print("cc", msg.control, msg.value, "duration", msg.time)

                if msg.type == "program_change":
                    timbre_register[idx] = msg.program
                    #print("channel", idx, "pc", msg.program, "time", time_counter, "duration", msg.time)

                if msg.type == "note_on":
                    
                    # print("on ", msg.note, "time", time_counter, "duration", msg.time, "velocity", msg.velocity)
                    note_on_start_time = time_counter // sr
                    note_on_end_time = (time_counter + msg.time) // sr
                    intensity = volume * msg.velocity // 127

					# When a note_on event *ends* the note start to be play 
					# Record end time of note_on event if there is no value in register
					# When note_off event happens, we fill in the color
                    if note_register[msg.note] == -1:
                        if noteidx == rednoteidx:
                            note_register[msg.note] = (note_on_end_time,100)
                        else:
                            note_register[msg.note] = (note_on_end_time,10)
                    else:
					# When note_on event happens again, we also fill in the color
                        old_end_time = note_register[msg.note][0]
                        if noteidx == rednoteidx:
                            roll[idx, msg.note, old_end_time: note_on_end_time] = 100
                            note_register[msg.note] = (note_on_end_time,100)
                        else:
                            roll[idx, msg.note, old_end_time: note_on_end_time] = 10
                            note_register[msg.note] = (note_on_end_time,10)
                        
                    noteidx +=1

                if msg.type == "note_off":
                    #print("off", msg.note, "time", time_counter, "duration", msg.time, "velocity", msg.velocity)
                    note_off_start_time = time_counter // sr
                    note_off_end_time = (time_counter + msg.time) // sr
                    note_on_end_time = note_register[msg.note][0]
                    intensity = note_register[msg.note][1]
					# fill in color
                    
                    roll[idx, msg.note, note_on_end_time:note_off_end_time] = intensity
                    if intensity == 100: print(idx, msg.note, note_on_end_time, note_off_end_time)
                    note_register[msg.note] = -1  # reinitialize register

                time_counter += msg.time

                # TODO : velocity -> done, but not verified
                # TODO: Pitch wheel
                # TODO: Channel - > Program Changed / Timbre catagory
                # TODO: real time scale of roll

            # if there is a note not closed at the end of a channel, close it
            for key, data in enumerate(note_register):
                if data != -1:
                    note_on_end_time = data[0]
                    intensity = data[1]
                    # print(key, note_on_end_time)
                    note_off_start_time = time_counter // sr
                    roll[idx, key, note_on_end_time:] = intensity
                note_register[idx] = -1

        return roll

    def draw_roll(self,rednoteidx):

        roll = self.get_roll(rednoteidx=rednoteidx)

        # build and set fig obj
        plt.ioff()
        fig = plt.figure(figsize=(4, 3))
        a1 = fig.add_subplot(111)
        a1.axis("equal")
        a1.set_facecolor("black")

        # change unit of time axis from tick to second
        tick = self.get_total_ticks()
        second = mido.tick2second(tick, self.ticks_per_beat, self.get_tempo())
        # print(second)
        if second > 10:
            x_label_period_sec = second // 10
        else:
            x_label_period_sec = second / 10  # ms
        # print(x_label_period_sec)
        x_label_interval = mido.second2tick(x_label_period_sec, self.ticks_per_beat, self.get_tempo()) / self.sr
        # print(x_label_interval)
        plt.xticks([int(x * x_label_interval) for x in range(20)], [round(x * x_label_period_sec, 1) for x in range(20)])

        # change scale and label of y axis
        plt.yticks([y*16 for y in range(8)], [y*16 for y in range(8)])

        # build colors
        cmap = mpl.colors.ListedColormap(['black','white', 'red'])
        bounds=[-1,1,50,127]  #0: 검정 10: 하양 100: 빨강 
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # draw piano roll and stack image on a1
        a1.imshow(roll[0], origin="lower", interpolation='nearest', cmap=cmap, aspect='auto', norm=norm)  #roll[0]: 차피 채널 1개밖에 없음
        for i in roll[0]:
            for j in i:
                if j != 10 and j!= 100 and j != 0: print(j)
        # show piano roll
        plt.draw()
        plt.ion()
        #plt.show()
        st.pyplot(fig)

    def get_tempo(self):
        try:
            return self.meta["set_tempo"]["tempo"]
        except:
            return 500000

    def get_total_ticks(self):
        max_ticks = 0
        for channel in range(16):
            ticks = sum(msg.time for msg in self.events[channel])
            if ticks > max_ticks:
                max_ticks = ticks
        return max_ticks


if __name__ == "__main__":
    mid = MidiFile("test_file/1.mid")

    # get the list of all events
    # events = mid.get_events()

    # get the np array of piano roll image
    roll = mid.get_roll()

    # draw piano roll by pyplot
    mid.draw_roll()


