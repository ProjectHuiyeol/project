create database song_db;  
use song_db;

create table if not exists `song_db`.`song`(
	`song_id` int not null,
    `song_title` varchar(1000) not null,
    primary key(`song_id`)
) engine = innoDB default character set = utf8mb4;

create table if not exists `song_db`.`playlist_info`(
	`playlist_id` int not null auto_increment,
    `playlist_name` varchar(384) not null,
    `playlist_ratio` double not null,
    primary key(`playlist_id`,`playlist_name`)
) engine = innoDB default character set = utf8mb4;

create table if not exists `song_db`.`playlist`( 
    `song_seq` int not null auto_increment,
    `playlist_id` int not null,
    `song_id` int not null,
    primary key(`song_seq`),
    foreign key (`song_id`) references `song_db`.`song`(`song_id`),
    foreign key(`playlist_id`) references `song_db`.`playlist_info`(`playlist_id`)
    on delete cascade
) engine = innoDB default character set = utf8mb4;

create table if not exists `song_db`.`users`(
    `playlist_seq` int not null auto_increment,
	`user_id` int not null, 
    `playlist_id` int not null,
    primary key (`playlist_seq`,`user_id`),
    foreign key(`playlist_id`) references `song_db`.`playlist_info`(`playlist_id`)
) engine = innoDB;

create table if not exists `song_db`.`song_info_sound`(
	`song_id` int not null,
    `song_mp3_path` varchar(1000) not null,
    `happy` double not null,
    `sad` double not null,
    `angry` double not null,
    `relaxed` double not null,
    primary key(`song_id`),
    foreign key(`song_id`) references `song_db`.`song`(`song_id`)
) engine  = innoDB default character set = utf8mb4;

create table if not exists `song_db`.`song_info_lyrics`(
	`song_id` int not null,
    `song_lyrics` varchar(8000) not null,
    `happy` double not null,
    `fear` double not null,
    `angry` double not null,
    `dislike` double not null,
    `surprise` double not null,
    `sad` double not null,
    primary key(`song_id`),
    foreign key(`song_id`) references `song_db`.`song`(`song_id`)
) engine  = innoDB default character set = utf8mb4;