// Per-finger int (1..10) shared constants. Single source of truth so the
// piano roll, finger buttons, and any future viewer cannot drift out of sync.
//
// 1..5  = Left  hand (thumb..pinky)
// 6..10 = Right hand (thumb..pinky)

export const FINGER_INT_COLOR = {
  1: '#E53935',   //  L1 thumb   — red
  2: '#FB8C00',   //  L2 index   — orange
  3: '#FDD835',   //  L3 middle  — yellow
  4: '#7CB342',   //  L4 ring    — light green
  5: '#00897B',   //  L5 pinky   — teal
  6: '#00BCD4',   //  R1 thumb   — cyan
  7: '#1E88E5',   //  R2 index   — blue
  8: '#3949AB',   //  R3 middle  — indigo
  9: '#8E24AA',   //  R4 ring    — purple
 10: '#D81B60',   //  R5 pinky   — pink
};

export const INT_TO_LABEL = {
  1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4', 5: 'L5',
  6: 'R1', 7: 'R2', 8: 'R3', 9: 'R4', 10: 'R5',
};

export const FINGER_INT_NAME = {
  1: 'thumb',  2: 'index',  3: 'middle',  4: 'ring',  5: 'little',
  6: 'thumb',  7: 'index',  8: 'middle',  9: 'ring', 10: 'little',
};
