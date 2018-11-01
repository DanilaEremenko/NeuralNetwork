function arabian_letter_example

font_name = 'Times New Roman';

path = 'd:\Doc\!!Docs\!Database\Letters\';

i_start = hex2dec('68e');
i_stop = hex2dec('6ab');
ct = 1;

for i = i_start:i_stop
    
    symbol = char(i);
    
    I = uint8(255*ones(100,100));
    
    RGB = insertText(I,[20 20], symbol,'Font', font_name, 'FontSize', 50, 'BoxColor','white');
    
    imshow(RGB)
    imwrite(RGB, [path 'image_' num2str(ct) '.jpg']);
    ct = ct+1;
end;